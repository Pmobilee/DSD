import os
import random
from PIL import Image
import numpy as np
import tqdm

def get_all_image_paths(root_dir):
    image_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

def select_random_images(image_paths, num_images):
    return random.sample(image_paths, min(num_images, len(image_paths)))

def load_images_as_arrays(image_paths):
    images = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(np.array(img))
    return images

def save_images_to_npz(images, filename):
    np.savez(filename, arr_0=np.array(images))

def process_subfolders(root_dir):
        # print(subdir_path)

    
            # print(sub_subdir)
    num_images = 30000  # Number of images you want to select
    image_paths = get_all_image_paths(root_dir)
    selected_images = select_random_images(image_paths, num_images)
    images_arrays = load_images_as_arrays(selected_images)
    name = root_dir.split('/')[-1]
    save_images_to_npz(images_arrays, f"{name}.npz")

# Example usage
root_dir = './archive/celeba_hq_256'
process_subfolders(root_dir)