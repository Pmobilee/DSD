import os
import shutil
import random

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def copy_random_images(src, dest, num_images):
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    all_images = []
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                all_images.append(os.path.join(root, file))

    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    for i, image in enumerate(selected_images):
        if i >= num_images:
            break
        dest_path = os.path.join(dest, os.path.basename(image))
        shutil.copy(image, dest_path)

def find_and_copy_images(root, num_images):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Temporarily remove numeric directories from dirnames
        numeric_dirs = [d for d in dirnames if is_number(d)]
        dirnames[:] = [d for d in dirnames if not is_number(d)]

        for dirname in numeric_dirs:
            number_dir_path = os.path.join(dirpath, dirname)
            loose_dir_path = os.path.join('./LOOSE', number_dir_path.strip('./'))
            copy_random_images(number_dir_path, loose_dir_path, num_images)

# Example usage
find_and_copy_images('./CelebA/celeb_original/', num_images=30000)
