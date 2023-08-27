import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # Load the pre-trained ResNet model
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Extract the first few layers (you can modify this according to the features you want)
        self.features = nn.Sequential(*list(resnet.children())[:5]) 
        print(f'Resnet architecture:\n{self.features}')

    def forward(self, x):
        return self.features(x)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Example usage
    input_image = torch.rand(1, 3, 480, 640).to(device) # Example input
    model = FeatureExtractor().to(device)
    output = model(input_image)
    print(output.shape) # Check the output shape