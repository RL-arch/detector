import os
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


def process_single_image(file, folder_end, output_folder, save_dir):
    img1 = Image.open(f"{output_folder}/img1/{file}")
    file_2 = file.replace("t02", "t13")
    img2 = Image.open(folder_end + file_2)
    file_3 = file.replace(".tif", "_diff.png")
    img3 = Image.open(f"{output_folder}/img3/{file_3}")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axs:
        ax.set_axis_off()

    axs[0].imshow(img1)
    axs[1].imshow(img2)
    axs[1].text(1, 15, "End", fontsize=5, weight="bold", color='k')
    axs[2].imshow(img3)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(f"{save_dir}/{file}", transparent=False, bbox_inches='tight', pad_inches=0.0, dpi=400)
    plt.close(fig)
    print(f'Saved at {save_dir}/{file}')


def final_image(folder_end, output_folder):
    save_dir = f"{output_folder}/final_image"
    os.makedirs(save_dir, exist_ok=True)

    # Get files for parallel processing
    files = sorted(os.listdir(f"{output_folder}/img1"), key=str.lower)

    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor() as executor:
        for file in files:
            executor.submit(process_single_image, file, folder_end, output_folder, save_dir)
