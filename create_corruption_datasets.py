import os
import shutil
from tqdm import tqdm
import random

random.seed(0)
if __name__ == "__main__":
    data_root = "./test_data"
    datasets = ['CoSal2015', 'CoSOD3k']
    corruption = ['dark5', 'brightness5', 'fog5', 'frost5', 'snow5']
    for ds in datasets:
        data_path = os.path.join(data_root, ds, 'Image')
        save_root = os.path.join(data_root, ds, 'Corruption')
        if not (os.path.isdir(save_root)): os.makedirs(save_root)
        classes = os.listdir(data_path)
        for cls in tqdm(classes):
            image_names = os.listdir(os.path.join(data_path, cls))
            num_imgs = len(image_names)
            image_paths = [os.path.join(data_path, cls, image_name) for image_name in image_names]
            for idx, path in enumerate(image_paths):
                corrupt_idx = idx % 5
                corrupt_path = path.replace(f'/{ds}/Image/', f'/{ds}_{corruption[corrupt_idx]}/')
                save_dir = os.path.join(save_root, cls)
                if not (os.path.isdir(save_dir)): os.makedirs(save_dir)
                save_path = os.path.join(save_dir, path.rsplit("/")[-1])
                shutil.copy2(corrupt_path, save_path)
            # print(corrupt_img_paths)
