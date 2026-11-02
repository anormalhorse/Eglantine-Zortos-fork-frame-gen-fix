# NVIDIA Optical Flow frame generation (NX2)

Terzo backend di frame generation, accanto a Lossless Scaling (LSFG) e FSR 3.
Usa l'Optical Flow Accelerator hardware delle GPU NVIDIA RTX tramite `VK_NV_optical_flow`.
Non usa DLL: il DLSS Frame Generation (`nvngx_dlssg.dll`) richiede motion vector e depth
del motore di gioco, che un emulatore non ha.

## Come si attiva
Configurazione > Graphics > Frame Generation Type > "NVIDIA Optical Flow (RTX 20+)".
Se la GPU/driver non lo supporta, o se qualcosa fallisce, il programma passa a FSR 3
e lo scrive nel log (`Render_Vulkan`).

## Come funziona
1. Il frame presentato viene copiato in una di due immagini (ping-pong).
2. L'acceleratore calcola il flusso tra frame nuovo e precedente su una coda dedicata
   (`VK_QUEUE_OPTICAL_FLOW_BIT_NV`), poi si aspetta la fine con una fence.
3. Il flusso (S10.5) viene copiato in un buffer e convertito da `nv_optical_flow_convert.comp`
   in un'immagine R16G16_SFLOAT in pixel.
4. I frame generati usano lo stesso shader di interpolazione di FSR 3.

## Note
- Mai provato su hardware: e' stato controllato solo che compili e che lo shader sia valido.
- Se i frame generati sembrano andare "all'indietro", avvia con la variabile d'ambiente
  `EDEN_NVOF_INVERT=1` (inverte il segno del flusso).
- Ogni frame fa una `scheduler.Finish()` per sincronizzare la coda dell'optical flow:
  semplice e sicuro, ma costa un po' di parallelismo CPU/GPU.
- Righe di log da cercare: "NVIDIA optical flow frame generation ready", "Optical flow queue family",
  "failed:".
