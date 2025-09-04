import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

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

# Testilataus

if __name__ == "__main__":
    content_path = os.path.join(CONTENT_DIR, "content.jpg")
    style_path = os.path.join(STYLE_DIR, "style.jpg")

    content_img = image_loader(content_path)
    style_img = image_loader(style_path)

    print("Content shape:", content_img.shape)
    print("Style shape:", style_img.shape)

with torch.no_grad():
    content_features = cnn(content_img)
    style_features = cnn(style_img)

print("VGG19 output contentille:", content_features.shape)
print("VGG19 output stylelle: ", style_features.shape)