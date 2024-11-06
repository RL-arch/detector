import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def resize_images(folder_images, img_size):
    """Resize images in `folder_images` to `img_size` if not already of the correct size."""
    img_size = (img_size, img_size)
    if not os.path.isdir(folder_images):
        print("Folder not found.")
        return

    def resize_file(img_path, size):
        """Check if an image needs resizing, and resize if necessary."""
        try:
            with Image.open(img_path) as img:
                if img.size != size:
                    img = img.resize(size)
                    img.save(img_path)  # Overwrite original file with resized image
                    print(f"Resized {img_path} to {size}")
                else:
                    print(f"Skipping {img_path} - already at target size {size}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Iterate over all folders and files, resizing as needed
    folders = sorted([f for f in os.listdir(folder_images) if f != ".DS_Store" and os.path.isdir(f"{folder_images}/{f}")])
    with ThreadPoolExecutor() as executor:
        for folder in folders:
            folder_path = os.path.join(folder_images, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".tif"):
                    img_path = os.path.join(folder_path, file)
                    executor.submit(resize_file, img_path, img_size)
    print(f"Checked and resized images to {img_size} where necessary.")



def rename(folder_images):
    if os.path.isdir(folder_images):
        i = 1
        folders = sorted([f for f in os.listdir(folder_images) if f != ".DS_Store" and os.path.isdir(f"{folder_images}/{f}")])

        def rename_folder(folder, index):
            try:
                shutil.move(f"{folder_images}/{folder}", f"{folder_images}/Experiment_{index}")
                print(f"Your folder {folder} is renamed as Experiment_{index}.")
            except PermissionError:
                print(f"Permission denied for folder {folder}, skipping.")
            except Exception as e:
                print(f"Error renaming {folder}: {e}")

        with ThreadPoolExecutor() as executor:
            for folder in folders:
                executor.submit(rename_folder, folder, i)
                i += 1

        print(f'Images are from {i - 1} experiment(s) in total')
    else:
        print("Folder not found.")
    return i - 1

def preprocess(folder_images, output_folder, num_exps):
    print("Preprocessing data structure...")

    os.makedirs(f"{output_folder}/start", exist_ok=True)
    os.makedirs(f"{output_folder}/end", exist_ok=True)

    def process_files(i):
        folder = f"{folder_images}/Experiment_{i}"
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if "t02" in file:
                    shutil.copyfile(f"{folder}/{file}", f"{output_folder}/start/{file}")
                elif "t12" in file:
                    shutil.copyfile(f"{folder}/{file}", f"{output_folder}/end/{file}")

    with ThreadPoolExecutor() as executor:
        executor.map(process_files, range(1, num_exps + 1))

    start, end = f"{output_folder}/start/", f"{output_folder}/end/"
    print("Done with preprocessing.")
    return start, end
