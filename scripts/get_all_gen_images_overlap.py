import os
import os.path as osp
import shutil

# existing images
images_base_path = "/mimer/NOBACKUP/groups/clippin/gen_images"
lidar_images_path = osp.join(images_base_path, "once_val_lidar")
img_images_path = osp.join(images_base_path, "once_val_img")
og_images_path = osp.join(images_base_path, "once_val_og")

in_foldernames = [lidar_images_path, img_images_path, og_images_path]

lidar_images = os.listdir(lidar_images_path)
img_images = os.listdir(img_images_path)
og_images = os.listdir(og_images_path)

# find overlap
overlapping_images = set(lidar_images).intersection(set(img_images)).intersection(set(og_images))

print(f"Found {len(overlapping_images)} overlapping images")

# move overlap to seperate folders
lidar_out_path = osp.join(images_base_path, "once_val_lidar_selected")
images_out_path = osp.join(images_base_path, "once_val_img_selected")
og_images_out_path = osp.join(images_base_path, "once_val_og_selected")
os.makedirs(lidar_out_path, exist_ok=True)
os.makedirs(images_out_path, exist_ok=True)
os.makedirs(og_images_out_path, exist_ok=True)

for i, folder_name_out in enumerate([lidar_out_path, images_out_path, og_images_out_path]):
    for filename in overlapping_images:
        filename_in = osp.join(in_foldernames[i], filename)
        filename_out = osp.join(folder_name_out, filename)
        shutil.move(filename_in, filename_out)
