import torch, sys
if not torch.cuda.is_available():
    sys.exit("GPU not detected. Remove bf16/fp16 flags.")
if not torch.cuda.is_bf16_supported():
    sys.exit("Current GPU/driver/PyTorch build lacks bf16 support.")
