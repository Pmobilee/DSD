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
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        # if "celeb_original" in subdir or "TSD" in subdir or "naive" in subdir or "exp" in subdir:
        #         continue
        print(subdir_path)
        if os.path.isdir(subdir_path):
            
            for size in tqdm.tqdm(['2', '4','8']):
                sub_subdir = os.path.join(subdir_path, size)
                if os.path.exists(sub_subdir):
                    # print(sub_subdir)
                    name = root_dir.split('/')[-1]
                    save_name = f"{name}_{subdir}_{size}.npz"

                    if os.path.exists(save_name):
                        continue
                    num_images = 30000  # Number of images you want to select
                    image_paths = get_all_image_paths(sub_subdir)
                    selected_images = select_random_images(image_paths, num_images)
                    images_arrays = load_images_as_arrays(selected_images)
                    
      
                    save_images_to_npz(images_arrays, save_name)

# Example usage
root_dir = '../saved_images/CelebA'
process_subfolders(root_dir)