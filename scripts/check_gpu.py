import torch

def check_gpu():
    if torch.cuda.is_available():
        device_id = 0
        gpu_name = torch.cuda.get_device_name(device_id)
        vram_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
        print(f"CUDA is available: True")
        print(f"GPU Name: {gpu_name}")
        print(f"VRAM: {vram_gb:.2f} GB")
    else:
        print("CUDA is available: False")
        print("PyTorch is running on CPU only.")

if __name__ == "__main__":
    check_gpu()
