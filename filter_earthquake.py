import os
import shutil

images_dir = 'images'
targets_dir = 'targets'

output_images_dir = 'dataset_earthquake_only/images'
output_targets_dir = 'dataset_earthquake_only/targets'
train_collapsed_dir = 'dataset_earthquake_only/train/collapsed'
train_intact_dir = 'dataset_earthquake_only/train/intact'

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_targets_dir, exist_ok=True)
os.makedirs(train_collapsed_dir, exist_ok=True)
os.makedirs(train_intact_dir, exist_ok=True)

for fname in os.listdir(images_dir):
    if 'earthquake' in fname:
        shutil.copy(os.path.join(images_dir, fname), os.path.join(output_images_dir, fname))

        target_name = fname.replace('.png', '_target.png')
        target_path = os.path.join(targets_dir, target_name)
        if os.path.exists(target_path):
            shutil.copy(target_path, os.path.join(output_targets_dir, target_name))

print("✅ Earthquake images and targets copied.")
