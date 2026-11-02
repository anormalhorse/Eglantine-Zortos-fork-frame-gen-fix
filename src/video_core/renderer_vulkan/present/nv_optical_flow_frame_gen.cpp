// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <vector>

#include "common/logging.h"
#include "video_core/renderer_vulkan/present/nv_optical_flow_frame_gen.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_common/vulkan_device.h"

#include "video_core/host_shaders/fsr3_frame_interpolation_comp_spv.h"
#include "video_core/host_shaders/nv_optical_flow_convert_comp_spv.h"

namespace Vulkan {

namespace {

// VK_FORMAT_R16G16_SFIXED5_NV (formerly VK_FORMAT_R16G16_S10_5_NV). Spelled numerically so the
// code builds with Vulkan headers that only know one of the two names.
constexpr VkFormat FLOW_VECTOR_FORMAT = static_cast<VkFormat>(1000464000);

// Maximum number of generated frames per presented frame (see Settings::MAX_FRAME_GEN_MULTIPLIER)
// plus one spare, so every generation of a frame gets its own descriptor set.
constexpr size_t INTERP_SET_COUNT = 4;

constexpr u64 OPTICAL_FLOW_TIMEOUT_NS = 2'000'000'000;

struct ConvertPushConstants {
    s32 flow_width;
    s32 flow_height;
    float scale;
};

struct InterpPushConstants {
    float inv_render_w;
    float inv_render_h;
    float render_w;
    float render_h;
    float blend_factor;
    float flow_scale;
};

[[nodiscard]] VkFormat InputFormatFor(VkFormat frame_format) {
    switch (frame_format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
        return VK_FORMAT_B8G8R8A8_UNORM;
    default:
        return VK_FORMAT_UNDEFINED;
    }
}

[[nodiscard]] VkOpticalFlowGridSizeFlagsNV GridSizeBit(u32 size) {
    switch (size) {
    case 1:
        return VK_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_NV;
    case 2:
        return VK_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_NV;
    case 8:
        return VK_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_NV;
    default:
        return VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV;
    }
}

[[nodiscard]] bool EnvFlag(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
        return false;
    }
    const bool enabled = value[0] == '1';
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
#endif
}

} // Anonymous namespace

bool NvOpticalFlowFrameGen::IsSupported(const Device& device, VkFormat frame_format) {
    return device.CanUseOpticalFlow() && InputFormatFor(frame_format) != VK_FORMAT_UNDEFINED;
}

NvOpticalFlowFrameGen::NvOpticalFlowFrameGen(const Device& device,
                                             MemoryAllocator& memory_allocator, VkExtent2D extent,
                                             VkFormat format)
    : m_device{device}, m_allocator{memory_allocator}, m_extent{extent}, m_format{format} {
    try {
        Initialize(device);
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "NVIDIA optical flow frame generation: initialization threw: {}",
                  e.what());
        m_failed = true;
    }
}

NvOpticalFlowFrameGen::~NvOpticalFlowFrameGen() {
    const auto& dld = m_device.GetDispatchLoader();
    const VkDevice dev = *m_device.GetLogical();
    if (m_session != VK_NULL_HANDLE && m_fn.destroy_session) {
        m_fn.destroy_session(dev, m_session, nullptr);
    }
    if (m_fence != VK_NULL_HANDLE) {
        dld.vkDestroyFence(dev, m_fence, nullptr);
    }
    if (m_command_pool != VK_NULL_HANDLE) {
        dld.vkDestroyCommandPool(dev, m_command_pool, nullptr);
    }
}

void NvOpticalFlowFrameGen::Fail(const char* reason) {
    if (!m_failed) {
        LOG_ERROR(Render_Vulkan,
                  "NVIDIA optical flow frame generation failed: {}. Falling back to FSR3.",
                  reason);
    }
    m_failed = true;
}

void NvOpticalFlowFrameGen::Initialize(const Device& device) {
    m_input_format = InputFormatFor(m_format);
    if (m_input_format == VK_FORMAT_UNDEFINED) {
        Fail("the presented frame format is not supported by the optical flow accelerator");
        return;
    }
    if (!device.CanUseOpticalFlow()) {
        Fail("VK_NV_optical_flow or its queue is not available");
        return;
    }

    // Debug aid: if generated frames look like they move backwards, set EDEN_NVOF_INVERT=1.
    m_flow_sign = EnvFlag("EDEN_NVOF_INVERT") ? -1.0f : 1.0f;

    if (!LoadFunctions(device)) {
        Fail("the optical flow entry points could not be loaded");
        return;
    }
    CreateQueueObjects(device);
    if (m_failed || !CreateSession(device)) {
        return;
    }
    CreateImages(device);
    if (m_failed) {
        return;
    }
    CreatePipelines(device);
    if (m_failed) {
        return;
    }

    const VkFormatProperties3 flow_properties =
        device.GetPhysical().GetFormatProperties3(FLOW_VECTOR_FORMAT);
    LOG_INFO(Render_Vulkan,
             "NVIDIA optical flow frame generation ready: {}x{} input, grid {}x{} -> {}x{} flow, "
             "flow format features {:#x}",
             m_extent.width, m_extent.height, m_grid_size, m_grid_size, m_flow_extent.width,
             m_flow_extent.height, static_cast<u64>(flow_properties.optimalTilingFeatures));
}

bool NvOpticalFlowFrameGen::LoadFunctions(const Device& device) {
    const auto& dld = device.GetDispatchLoader();
    const VkDevice dev = *device.GetLogical();

    m_fn.create_session = reinterpret_cast<PFN_vkCreateOpticalFlowSessionNV>(
        dld.vkGetDeviceProcAddr(dev, "vkCreateOpticalFlowSessionNV"));
    m_fn.destroy_session = reinterpret_cast<PFN_vkDestroyOpticalFlowSessionNV>(
        dld.vkGetDeviceProcAddr(dev, "vkDestroyOpticalFlowSessionNV"));
    m_fn.bind_image = reinterpret_cast<PFN_vkBindOpticalFlowSessionImageNV>(
        dld.vkGetDeviceProcAddr(dev, "vkBindOpticalFlowSessionImageNV"));
    m_fn.execute = reinterpret_cast<PFN_vkCmdOpticalFlowExecuteNV>(
        dld.vkGetDeviceProcAddr(dev, "vkCmdOpticalFlowExecuteNV"));

    return m_fn.create_session && m_fn.destroy_session && m_fn.bind_image && m_fn.execute;
}

void NvOpticalFlowFrameGen::CreateQueueObjects(const Device& device) {
    const auto& dld = device.GetDispatchLoader();
    const VkDevice dev = *device.GetLogical();

    const VkCommandPoolCreateInfo pool_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.GetOpticalFlowFamily(),
    };
    if (dld.vkCreateCommandPool(dev, &pool_ci, nullptr, &m_command_pool) != VK_SUCCESS) {
        Fail("could not create the optical flow command pool");
        return;
    }

    const VkCommandBufferAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = m_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (dld.vkAllocateCommandBuffers(dev, &allocate_info, &m_command_buffer) != VK_SUCCESS) {
        Fail("could not allocate the optical flow command buffer");
        return;
    }

    const VkFenceCreateInfo fence_ci{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    if (dld.vkCreateFence(dev, &fence_ci, nullptr, &m_fence) != VK_SUCCESS) {
        Fail("could not create the optical flow fence");
    }
}

bool NvOpticalFlowFrameGen::CreateSession(const Device& device) {
    const VkDevice dev = *device.GetLogical();

    // Prefer the finest grid that is still cheap; fall back to coarser ones (and finally 1x1)
    // when the hardware does not support it.
    constexpr std::array<u32, 4> GRID_CANDIDATES{4, 8, 2, 1};
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    for (const u32 grid : GRID_CANDIDATES) {
        const VkOpticalFlowSessionCreateInfoNV session_ci{
            .sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV,
            .pNext = nullptr,
            .width = m_extent.width,
            .height = m_extent.height,
            .imageFormat = m_input_format,
            .flowVectorFormat = FLOW_VECTOR_FORMAT,
            .costFormat = VK_FORMAT_UNDEFINED,
            .outputGridSize = GridSizeBit(grid),
            .hintGridSize = 0,
            .performanceLevel = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV,
            .flags = 0,
        };
        result = m_fn.create_session(dev, &session_ci, nullptr, &m_session);
        if (result == VK_SUCCESS) {
            m_grid_size = grid;
            break;
        }
        LOG_WARNING(Render_Vulkan,
                    "Optical flow session with grid {}x{} was rejected (VkResult {})", grid, grid,
                    static_cast<s32>(result));
        m_session = VK_NULL_HANDLE;
    }
    if (result != VK_SUCCESS) {
        Fail("the optical flow session could not be created");
        return false;
    }

    m_flow_extent = VkExtent2D{
        .width = (m_extent.width + m_grid_size - 1) / m_grid_size,
        .height = (m_extent.height + m_grid_size - 1) / m_grid_size,
    };
    return true;
}

void NvOpticalFlowFrameGen::CreateImages(const Device& device) {
    // The accelerator runs on a different queue family than the one that fills the images, so
    // the shared images are created concurrent and no ownership transfers are needed.
    const std::array<u32, 2> families{device.GetGraphicsFamily(), device.GetOpticalFlowFamily()};

    // Images used by the accelerator declare how through VkOpticalFlowImageFormatInfoNV in the
    // pNext chain (there is no VkImageUsageFlagBits for it).
    const auto make_image = [&](VkExtent2D extent, VkFormat format, VkImageUsageFlags usage,
                                bool concurrent, VkOpticalFlowUsageFlagsNV optical_flow_usage) {
        const VkOpticalFlowImageFormatInfoNV optical_flow_info{
            .sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV,
            .pNext = nullptr,
            .usage = optical_flow_usage,
        };
        const VkImageCreateInfo image_ci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = optical_flow_usage != 0 ? &optical_flow_info : nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {.width = extent.width, .height = extent.height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .sharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = concurrent ? static_cast<u32>(families.size()) : 0u,
            .pQueueFamilyIndices = concurrent ? families.data() : nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        return m_allocator.CreateImage(image_ci);
    };

    for (size_t i = 0; i < m_input_image.size(); ++i) {
        m_input_image[i] = make_image(m_extent, m_input_format,
                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                      true, VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV);
        m_input_view[i] = CreateWrappedImageView(device, m_input_image[i], m_input_format);
    }

    m_flow_image = make_image(m_flow_extent, FLOW_VECTOR_FORMAT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              true, VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV);
    m_flow_view = CreateWrappedImageView(device, m_flow_image, FLOW_VECTOR_FORMAT);

    // Motion in full-resolution pixels, sampled by the interpolation shader.
    m_motion_image = make_image(m_flow_extent, VK_FORMAT_R16G16_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, false,
                                0);
    m_motion_view = CreateWrappedImageView(device, m_motion_image, VK_FORMAT_R16G16_SFLOAT);

    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = static_cast<VkDeviceSize>(m_flow_extent.width) * m_flow_extent.height * 4,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    m_flow_buffer = m_allocator.CreateBuffer(buffer_ci, MemoryUsage::DeviceLocal);

    // The flow vector image stays bound for the whole life of the session.
    const VkResult bind_result = m_fn.bind_image(
        *device.GetLogical(), m_session, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV,
        *m_flow_view, VK_IMAGE_LAYOUT_GENERAL);
    if (bind_result != VK_SUCCESS) {
        Fail("the flow vector image could not be bound to the session");
    }
}

void NvOpticalFlowFrameGen::CreatePipelines(const Device& device) {
    const auto& logical = device.GetLogical();

    m_linear_sampler = CreateBilinearSampler(device);
    m_convert_shader = BuildShader(device, NV_OPTICAL_FLOW_CONVERT_COMP_SPV);
    m_interp_shader = BuildShader(device, FSR3_FRAME_INTERPOLATION_COMP_SPV);

    const std::array pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                             static_cast<u32>(3 * INTERP_SET_COUNT)},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                             static_cast<u32>(INTERP_SET_COUNT + 1)},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
    };
    m_desc_pool = logical.CreateDescriptorPool(VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = static_cast<u32>(INTERP_SET_COUNT + 1),
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });

    // Flow conversion: buffer -> motion image.
    const std::array convert_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    m_convert_desc_layout = logical.CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<u32>(convert_bindings.size()),
        .pBindings = convert_bindings.data(),
    });
    const VkPushConstantRange convert_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ConvertPushConstants),
    };
    m_convert_pipe_layout = logical.CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_convert_desc_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &convert_push_range,
    });
    m_convert_pipeline = logical.CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *m_convert_shader,
            .pName = "main",
        },
        .layout = *m_convert_pipe_layout,
    });

    // Frame interpolation (shared with the FSR3 backend).
    const std::array interp_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    m_interp_desc_layout = logical.CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<u32>(interp_bindings.size()),
        .pBindings = interp_bindings.data(),
    });
    const VkPushConstantRange interp_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(InterpPushConstants),
    };
    m_interp_pipe_layout = logical.CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_interp_desc_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &interp_push_range,
    });
    m_interp_pipeline = logical.CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *m_interp_shader,
            .pName = "main",
        },
        .layout = *m_interp_pipe_layout,
    });

    m_convert_desc_sets = CreateWrappedDescriptorSets(m_desc_pool, {*m_convert_desc_layout});

    std::array<VkDescriptorSetLayout, INTERP_SET_COUNT> interp_layouts;
    interp_layouts.fill(*m_interp_desc_layout);
    m_interp_desc_sets = CreateWrappedDescriptorSets(m_desc_pool, interp_layouts);

    // The conversion set never changes: flow buffer in, motion image out.
    const VkDescriptorBufferInfo buffer_info{
        .buffer = *m_flow_buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    const VkDescriptorImageInfo motion_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = *m_motion_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const std::array writes{
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_convert_desc_sets[0],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_info,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_convert_desc_sets[0],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &motion_info,
        },
    };
    logical.UpdateDescriptorSets(writes, {});
}

bool NvOpticalFlowFrameGen::RunOpticalFlow(size_t input_slot, size_t reference_slot) {
    const auto& dld = m_device.GetDispatchLoader();
    const VkDevice dev = *m_device.GetLogical();

    // input = newest frame, reference = previous frame: the field describes how content moved
    // from the previous frame to the newest one, which is what the interpolation shader expects.
    VkResult result = m_fn.bind_image(dev, m_session, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV,
                                      *m_input_view[input_slot], VK_IMAGE_LAYOUT_GENERAL);
    if (result != VK_SUCCESS) {
        Fail("the input image could not be bound");
        return false;
    }
    result = m_fn.bind_image(dev, m_session, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV,
                             *m_input_view[reference_slot], VK_IMAGE_LAYOUT_GENERAL);
    if (result != VK_SUCCESS) {
        Fail("the reference image could not be bound");
        return false;
    }

    // Beginning a command buffer from a pool created with RESET_COMMAND_BUFFER implicitly resets it.
    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    if (dld.vkBeginCommandBuffer(m_command_buffer, &begin_info) != VK_SUCCESS) {
        Fail("could not begin the optical flow command buffer");
        return false;
    }

    const VkOpticalFlowExecuteInfoNV execute_info{
        .sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV,
        .pNext = nullptr,
        .flags = 0,
        .regionCount = 0,
        .pRegions = nullptr,
    };
    m_fn.execute(m_command_buffer, m_session, &execute_info);

    if (dld.vkEndCommandBuffer(m_command_buffer) != VK_SUCCESS) {
        Fail("could not end the optical flow command buffer");
        return false;
    }

    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &m_command_buffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    if (m_device.GetOpticalFlowQueue().Submit(submit_info, m_fence) != VK_SUCCESS) {
        Fail("could not submit to the optical flow queue");
        return false;
    }

    // The graphics queue is idle for this frame's copy (the caller finished the scheduler), and
    // the converted result is recorded right after this, so a host wait keeps things simple.
    result = dld.vkWaitForFences(dev, 1, &m_fence, VK_TRUE, OPTICAL_FLOW_TIMEOUT_NS);
    if (result != VK_SUCCESS) {
        Fail("the optical flow queue did not finish in time");
        return false;
    }
    if (dld.vkResetFences(dev, 1, &m_fence) != VK_SUCCESS) {
        Fail("could not reset the optical flow fence");
        return false;
    }
    return true;
}

void NvOpticalFlowFrameGen::RecordFlowConversion(Scheduler& scheduler) {
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier to_transfer{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, to_transfer);

        const VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {m_flow_extent.width, m_flow_extent.height, 1},
        };
        cmdbuf.CopyImageToBuffer(*m_flow_image, VK_IMAGE_LAYOUT_GENERAL, *m_flow_buffer, region);

        const VkMemoryBarrier to_compute{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, to_compute);

        const ConvertPushConstants constants{
            .flow_width = static_cast<s32>(m_flow_extent.width),
            .flow_height = static_cast<s32>(m_flow_extent.height),
            .scale = m_flow_sign / 32.0f, // S10.5 fixed point
        };
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_convert_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_convert_pipe_layout, 0,
                                  m_convert_desc_sets[0], {});
        cmdbuf.PushConstants(*m_convert_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((m_flow_extent.width + 15) / 16, (m_flow_extent.height + 15) / 16, 1);

        const VkMemoryBarrier to_interpolation{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, to_interpolation);
    });
}

void NvOpticalFlowFrameGen::Process(const Device& device, Scheduler& scheduler,
                                    VkImage current_frame, VkImageView /*current_view*/,
                                    u64 frame_count) {
    (void)device;
    if (m_failed) {
        return;
    }
    if (m_has_frame && frame_count == m_last_frame_count && m_flow_valid) {
        return;
    }
    m_last_frame_count = frame_count;

    const bool first = !m_has_frame;
    const size_t slot = first ? 0 : (m_current_slot ^ 1);
    const bool initialize_layouts = !m_layouts_initialized;
    m_layouts_initialized = true;

    // Copy the presented frame into the accelerator input image. All the images involved stay in
    // GENERAL, which is the layout VK_NV_optical_flow requires for the images it uses.
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, current_frame, slot, initialize_layouts](vk::CommandBuffer cmdbuf) {
        if (initialize_layouts) {
            const auto to_general = [](VkImage image) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = image,
                    .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };
            };
            const std::array barriers{
                to_general(*m_input_image[0]),
                to_general(*m_input_image[1]),
                to_general(*m_flow_image),
                to_general(*m_motion_image),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, {}, {}, barriers);
        }

        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);

        const VkImageCopy region{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {m_extent.width, m_extent.height, 1},
        };
        cmdbuf.CopyImage(current_frame, VK_IMAGE_LAYOUT_GENERAL, *m_input_image[slot],
                         VK_IMAGE_LAYOUT_GENERAL, region);

        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, after);
    });

    if (first) {
        m_has_frame = true;
        m_current_slot = slot;
        return;
    }

    // Make sure the copy above reached the GPU before the accelerator reads the image.
    scheduler.Finish();

    if (!RunOpticalFlow(slot, m_current_slot)) {
        return;
    }
    RecordFlowConversion(scheduler);

    m_current_slot = slot;
    m_flow_valid = true;
}

void NvOpticalFlowFrameGen::Generate(const Device& device, Scheduler& scheduler,
                                     VkImageView destination_storage_view, float blend_factor,
                                     float /*flow_scale*/) {
    if (m_failed || !m_has_frame || !m_flow_valid) {
        return;
    }

    // Every generation of a frame gets its own descriptor set: they are all recorded before any
    // of them runs, so rewriting a single shared set would redirect the earlier dispatches.
    const size_t set_index = m_interp_ring++ % INTERP_SET_COUNT;
    const VkDescriptorSet set = m_interp_desc_sets[set_index];

    std::vector<VkDescriptorImageInfo> image_infos;
    image_infos.reserve(4);
    std::vector<VkWriteDescriptorSet> updates{
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler, *m_input_view[m_current_slot],
                                 set, 0),
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler,
                                 *m_input_view[m_current_slot ^ 1], set, 1),
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler, *m_motion_view, set, 2),
    };
    const VkDescriptorImageInfo destination_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = destination_storage_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    updates.push_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = 3,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &destination_info,
    });
    device.GetLogical().UpdateDescriptorSets(updates, {});

    // The motion image is already in full-resolution pixels, hence a flow scale of 1.
    const u32 out_w = m_extent.width;
    const u32 out_h = m_extent.height;
    const InterpPushConstants constants{
        .inv_render_w = 1.0f / static_cast<float>(out_w),
        .inv_render_h = 1.0f / static_cast<float>(out_h),
        .render_w = static_cast<float>(out_w),
        .render_h = static_cast<float>(out_h),
        .blend_factor = blend_factor,
        .flow_scale = 1.0f,
    };

    scheduler.Record([this, constants, set, out_w, out_h](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_interp_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_interp_pipe_layout, 0, set,
                                  {});
        cmdbuf.PushConstants(*m_interp_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
    });
}

} // namespace Vulkan
