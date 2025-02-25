import torch
import torchvision

print(torch.__version__)      # Should be 2.6.0+cu118
print(torchvision.__version__) # Should be 0.17.0 (or compatible version)
print(torch.cuda.is_available())  # Should return True
