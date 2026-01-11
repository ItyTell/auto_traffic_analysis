import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Пристрій: {torch.cuda.get_device_name(0)}")