import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn

# Kansiorakenne

CONTENT_DIR = "content_images"
STYLE_DIR = "style_images"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kuvien käsittely

loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def image_loader(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

# Ladataan malli (VGG19)

cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()

# Content loss

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0
    
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

#Rakennetaan uusi malli, johon lisätään content loss tiettyyn kohtaan

def get_content_model_and_losses(cnn, content_img, content_layers=['conv_4']):
    model = nn.Sequential()
    content_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f"Tuntematon kerros: {layer.__class__.__name__}")
        
        model.add_module(name, layer)

        # Jos tämä kerros on content loss -listalla

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
    
    return model, content_losses


# Testilataus

if __name__ == "__main__":
    content_path = os.path.join(CONTENT_DIR, "content.jpg")
    style_path = os.path.join(STYLE_DIR, "style.jpg")

    content_img = image_loader(content_path)

    model, content_losses = get_content_model_and_losses(cnn, content_img)

    print("Mallin rakenne:")
    print(model)

    print("\nContent loss target shape:")
    for cl in content_losses:
        print(cl.target.shape)