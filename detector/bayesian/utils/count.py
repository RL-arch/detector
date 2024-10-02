from tkinter import font
import torch
import os
from utils.bay_loss_utils.vgg import vgg19
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from torchvision import transforms
import pandas as pd


def docount(folder_start, model_path, folder_output):
    # Counting the number of cells in the image.
    print("Counting total amounts of organoids in the images...")
    # Load model for cell counting
    model = vgg19()
    device = torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Create the preprocessing transformation here
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    count_list = []
    if not os.path.exists(f"{folder_output}/img1"):
        os.mkdir(f"{folder_output}/img1")
    for file in sorted(os.listdir(folder_start), key=str.lower):
        filename = os.fsdecode(file)
        print(f"Processing file {filename}")

        img = cv2.imread(folder_start + filename)
        # Transform
        input = transform(img)

        # unsqueeze batch dimension, in case you are dealing with a single image
        input = input.unsqueeze(0)
        input = input.to(device)

        # Get prediction
        with torch.set_grad_enabled(False):
            output = model(input)
            count = (torch.sum(output).item())
            count_list.append(count)

        # create start and end image
        img = Image.open(folder_start + filename)
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        try:
            # Use 'Arial' as a default font which is generally available on most systems
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            # Fallback to the default font in case 'Arial' is not available
            font = ImageFont.load_default()
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((1, 5), "Start", (0, 0, 0), font=font)
        draw.text((1, 25), "Est.: {:.0f}".format(count), (0, 0, 0), font=font)

        save_dir = f"{folder_output}/img1/{filename}"
        img.save(save_dir)
    print("Image saved.")

    # Saving the result of the cell counting in an excel file.
    name_list = []
    excel_dir = f"{folder_output}/excel"
    os.makedirs(excel_dir, exist_ok=True)
    writer = pd.ExcelWriter(f"{excel_dir}/total amount.xlsx",
                            engine='xlsxwriter')
    name_list.extend(
        iter(sorted(os.listdir(f"{folder_output}/diff_images"),
                    key=str.lower)))

    df = pd.DataFrame({
        'Name': name_list,
        'Total': count_list,
    })
    df.to_excel(writer, index=False, header=True)
    # writer.save()
    writer.close()
    print(f"Result of total amount saved in {folder_output}/excel.")
    # return count_list