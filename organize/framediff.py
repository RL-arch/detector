from PIL import Image, ImageChops
import os
import cv2
import numpy as np


# Remove shadow
def remove_shadow(image):
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    # open_cv_image = np.asfarray(image)
    # Remove shadow
    rgb_planes = cv2.split(open_cv_image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((3, 3), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,
                                 None,
                                 alpha=0,
                                 beta=255,
                                 norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return Image.fromarray(result_norm)


def difference(folder_start, folder_end, output_folder, replace_from="t02", replace_to="t12"):
    print("Calculating frame differences...")
    if not os.path.exists(f"{output_folder}/diff_images"):
        os.mkdir(f"{output_folder}/diff_images")
    for file in os.listdir(folder_start):
        # hidden files in MacOS
        if file == (folder_start or folder_end or output_folder) + "/" + '.DS_Store':
            continue
        filename = os.fsdecode(file)
        filename_2 = filename.replace(replace_from, replace_to)

        # detect file extension
        extention = os.path.splitext(filename)[1]

        # assign images
        img1 = Image.open(os.path.join(folder_start, filename))
        img2 = Image.open(os.path.join(folder_end, filename_2))

        # remove shadow
        shadow_out_img1 = remove_shadow(img1)
        shadow_out_img2 = remove_shadow(img2)

        # finding difference
        diff = ImageChops.difference(shadow_out_img1, shadow_out_img2)

        # saving
        save_dir = os.path.join(output_folder, "diff_images", filename.replace(extention, "_diff.png"))

        try:
            diff.save(save_dir)
        except IOError as e:
            print(f"Error saving {save_dir}: {e}")
    print(f"Frame difference images are saved at {output_folder}/diff_images.")
