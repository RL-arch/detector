import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import shutil
from PIL import Image
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd




def finalize(output_dir_o):
    # process the final images
    folder_end = f"{output_dir_o}/end/"
    for file in sorted(os.listdir(f"{output_dir_o}/img1"), key=str.lower):
        img1 = Image.open(f"{output_dir_o}/img1/" + file)
        file_2 = file.replace("t02", "t12")
        img2 = Image.open(folder_end + file_2)
        file_3 = file.replace(".tif", "_diff.png")
        img3 = Image.open(f"{output_dir_o}/img3/" + file_3)

        fig = plt.figure()

        ax1 = fig.add_subplot(1, 3, 1)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        ax1.imshow(img1)

        ax2 = fig.add_subplot(1, 3, 2)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        plt.text(1, 15, "End", fontsize=5, weight="bold", color='k')
        ax2.imshow(img2)

        ax3 = fig.add_subplot(1, 3, 3)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        ax3.imshow(img3)

        save_dir = f"{output_dir_o}/final_image"
        os.makedirs(save_dir, exist_ok=True)
        print('Save at', save_dir + "/" + file)
        plt.savefig(save_dir + "/" + file,
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=400)
        plt.close()
        print("Final images saved! ")

    shutil.rmtree(f"{output_dir_o}/start")
    shutil.rmtree(f"{output_dir_o}/end")
    shutil.rmtree(f"{output_dir_o}/img1")
    shutil.rmtree(f"{output_dir_o}/diff_images")
    shutil.rmtree(f"{output_dir_o}/img3")
    print(f"cache in {output_dir_o} released.")

    # process the final excel
    df1 = pd.read_excel(f'{output_dir_o}/excel/total amount.xlsx')
    df2 = pd.read_excel(f'{output_dir_o}/excel/swelling amount.xlsx')

    if df1.empty or df2.empty:
        merged = df1.assign(Count=0)
    else:
        merged = df1.merge(df2, on="Name", how="left")
        merged.fillna(0, inplace=True)

    writer = pd.ExcelWriter(f"{output_dir_o}/excel/final results.xlsx",
                            engine='xlsxwriter')

    merged.to_excel(writer, index=False, header=True)
    # writer.save()
    writer.close()
    print("Final Excel saved as 'final results.xlsx'.")
