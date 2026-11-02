// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Frame generation backed by the NVIDIA Optical Flow Accelerator (VK_NV_optical_flow).
///
/// The hardware computes the motion field between the last two presented frames on its own
/// queue. The field is converted to a float motion image and fed to the same interpolation shader
/// used by the FSR3 backend. Needs an NVIDIA RTX GPU (Turing or newer) and a recent driver.
///
/// Every failure is reported through HasFailed(); the caller is expected to fall back to another
/// backend when it is set.
class NvOpticalFlowFrameGen {
public:
    /// True when the device exposes VK_NV_optical_flow with its queue and the presented frame
    /// format can be handed to the accelerator.
    [[nodiscard]] static bool IsSupported(const Device& device, VkFormat frame_format);

    explicit NvOpticalFlowFrameGen(const Device& device, MemoryAllocator& memory_allocator,
                                   VkExtent2D extent, VkFormat format);
    ~NvOpticalFlowFrameGen();

    NvOpticalFlowFrameGen(const NvOpticalFlowFrameGen&) = delete;
    NvOpticalFlowFrameGen& operator=(const NvOpticalFlowFrameGen&) = delete;

    void Process(const Device& device, Scheduler& scheduler, VkImage current_frame,
                 VkImageView current_view, u64 frame_count);

    void Generate(const Device& device, Scheduler& scheduler, VkImageView destination_storage_view,
                  float blend_factor, float flow_scale);

    [[nodiscard]] bool IsReady() const {
        return !m_failed && m_has_frame && m_flow_valid;
    }

    [[nodiscard]] bool HasFailed() const {
        return m_failed;
    }

private:
    struct Functions {
        PFN_vkCreateOpticalFlowSessionNV create_session{};
        PFN_vkDestroyOpticalFlowSessionNV destroy_session{};
        PFN_vkBindOpticalFlowSessionImageNV bind_image{};
        PFN_vkCmdOpticalFlowExecuteNV execute{};
    };

    void Initialize(const Device& device);
    bool LoadFunctions(const Device& device);
    void CreateImages(const Device& device);
    bool CreateSession(const Device& device);
    void CreateQueueObjects(const Device& device);
    void CreatePipelines(const Device& device);
    void Fail(const char* reason);

    [[nodiscard]] bool RunOpticalFlow(size_t input_slot, size_t reference_slot);
    void RecordFlowConversion(Scheduler& scheduler);

    const Device& m_device;
    MemoryAllocator& m_allocator;
    VkExtent2D m_extent;
    VkExtent2D m_flow_extent{};
    VkFormat m_format;
    VkFormat m_input_format{VK_FORMAT_UNDEFINED};
    u32 m_grid_size{4};
    float m_flow_sign{1.0f};

    Functions m_fn{};
    VkOpticalFlowSessionNV m_session{VK_NULL_HANDLE};
    VkCommandPool m_command_pool{VK_NULL_HANDLE};
    VkCommandBuffer m_command_buffer{VK_NULL_HANDLE};
    VkFence m_fence{VK_NULL_HANDLE};

    // Ping-pong pair: the accelerator input/reference and the interpolation sources.
    std::array<vk::Image, 2> m_input_image;
    std::array<vk::ImageView, 2> m_input_view;

    vk::Image m_flow_image;
    vk::ImageView m_flow_view;
    vk::Buffer m_flow_buffer;

    vk::Image m_motion_image;
    vk::ImageView m_motion_view;

    vk::Sampler m_linear_sampler;
    vk::ShaderModule m_convert_shader;
    vk::ShaderModule m_interp_shader;

    vk::DescriptorSetLayout m_convert_desc_layout;
    vk::PipelineLayout m_convert_pipe_layout;
    vk::Pipeline m_convert_pipeline;

    vk::DescriptorSetLayout m_interp_desc_layout;
    vk::PipelineLayout m_interp_pipe_layout;
    vk::Pipeline m_interp_pipeline;

    vk::DescriptorPool m_desc_pool;
    vk::DescriptorSets m_convert_desc_sets;
    vk::DescriptorSets m_interp_desc_sets; // one per ping-pong parity

    bool m_failed{};
    bool m_layouts_initialized{};
    bool m_has_frame{};
    bool m_flow_valid{};
    size_t m_current_slot{};
    size_t m_interp_ring{};
    u64 m_last_frame_count{};
};

} // namespace Vulkan
