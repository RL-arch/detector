import os
import shutil


def rename(folder_images):  # sourcery skip: raise-specific-error
    if os.path.isdir(folder_images):
        i = 1
        for folder in sorted(os.listdir(folder_images)):
            #hidden files in MacOS
            if folder == f"{folder_images}/.DS_Store":
                continue
            if os.path.isdir(f"{folder_images}/{folder}"):
                try:
                    #remaming the folders
                    shutil.move(f"{folder_images}/{folder}",
                                f"{folder_images}/" + f'Experiment_{i}')

                    print(f"Your folder {folder} is renamed as Experiment_{i}.")
                    i += 1
                except PermissionError:
                    continue
                except Exception as e:
                    raise Exception(e) from e

        print(f'Images are from {i - 1} experiment(s) in total')
    else:
        print("Folder not found.")
    return i - 1


def preprocess(folder_images, output_folder, num_exps):
    print("Preprocessing data structure...")
    for i in range(1, num_exps + 1):
        folder = f"{folder_images}/Experiment_{i}"
        if not os.path.exists(f"{output_folder}/start"):
            os.mkdir(f"{output_folder}/start")
        if not os.path.exists(f"{output_folder}/end"):
            os.mkdir(f"{output_folder}/end")
        for file in os.listdir(folder):
            if "t02" in file:
                original = f"{folder}/{file}"
                target = f"{output_folder}/start/{file}"
                shutil.copyfile(original, target)
            elif "t12" in file:
                original = f"{folder}/{file}"
                target = f"{output_folder}/end/{file}"
                shutil.copyfile(original, target)
    start = f"{output_folder}/start/"
    end = f"{output_folder}/end/"
    print("Done with preprocessing.")
    return start, end