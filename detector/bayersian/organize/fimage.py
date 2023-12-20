from matplotlib import pyplot as plt
import os
from PIL import Image


def final_image(folder_end, output_folder):
    for file in sorted(os.listdir(f"{output_folder}/img1"), key=str.lower):
        img1 = Image.open(f"{output_folder}/img1/{file}")
        # file_2 = file.replace("t2", "t6")
        file_2 = file.replace("t02", "t13")
        img2 = Image.open(folder_end + file_2)
        file_3 = file.replace(".tif", "_diff.png")
        img3 = Image.open(f"{output_folder}/img3/{file_3}")

        fig = plt.figure()

        ax1 = fig.add_subplot(1, 3, 1)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)
        ax1.imshow(img1)

        ax2 = fig.add_subplot(1, 3, 2)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)
        plt.text(1, 15, "End", fontsize=5, weight="bold", color='k')
        ax2.imshow(img2)

        ax3 = fig.add_subplot(1, 3, 3)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)
        ax3.imshow(img3)

        save_dir = f"{output_folder}/final_image"
        os.makedirs(save_dir, exist_ok=True)
        print('Save at', f"{save_dir}/{file}")
        plt.savefig(f"{save_dir}/{file}", transparent=False, bbox_inches='tight', pad_inches=0.0, dpi=400)

        plt.close()
        