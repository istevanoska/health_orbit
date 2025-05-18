import os
import shutil

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Clear folders before copying
clear_folder('dataset_earthquake_only/train/collapsed')
clear_folder('dataset_earthquake_only/train/intact')

images_dir = 'images'
collapsed_dir = 'dataset_earthquake_only/train/collapsed'
intact_dir = 'dataset_earthquake_only/train/intact'

for fname in os.listdir(images_dir):
    if 'earthquake' in fname:
        if 'post_disaster' in fname:
            shutil.copy(os.path.join(images_dir, fname), os.path.join(collapsed_dir, fname))
        elif 'pre_disaster' in fname:
            shutil.copy(os.path.join(images_dir, fname), os.path.join(intact_dir, fname))

print("✅ Only post_disaster images copied to collapsed folder")
print("✅ Only pre_disaster images copied to intact folder")
