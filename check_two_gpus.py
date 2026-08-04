import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

count = torch.cuda.device_count()
print("Visible GPU count:", count)

assert torch.cuda.is_available(), "CUDA is unavailable"
assert count == 2, f"Expected 2 GPUs, but found {count}"

for index in range(count):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    tensor = torch.ones(1, device=f"cuda:{index}")
    print(f"GPU {index} tensor:", tensor)

print("Two-GPU CUDA test passed.")
