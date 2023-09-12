import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.inception import InceptionScore
import os

cwd = os.getcwd()

path = f"{cwd}/to_test"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        if img is not None:
            if img.mode == 'RGB':
                transform = transforms.Compose([transforms.ToTensor()])
                tensor_img = transform(img)
                tensor_img = (tensor_img * 255).type(torch.uint8)
                images.append(tensor_img)
    tensor_images = torch.stack(images)
    return tensor_images

images = load_images_from_folder(path)
inception = InceptionScore(feature=2048)
inception.update(images)
print(inception.compute())