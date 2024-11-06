import os
import tkinter as tk
from tkinter import filedialog, messagebox

CONFIG_FILE = "config.txt"

# Function to read the config.txt file
def read_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            lines = f.readlines()
            return (
                lines[0].split('=')[1].strip().strip('"'),  # folder_images
                lines[1].split('=')[1].strip().strip('"'),  # output_folder
                lines[2].split('=')[1].strip().strip('"'),  # model_baylos
                lines[3].split('=')[1].strip().strip('"'),  # model_yolov7
            )
    else:
        return (
            "data/Input",
            "data/Output",
            "data/trained_models/model_count.pth",
            "data/trained_models/model_detect.pt"
        )

# Function to update the config.txt file
def update_config(folder_images, output_folder, model_baylos, model_yolov7):
    with open(CONFIG_FILE, "w") as f:
        f.write(f'folder_images = "{folder_images}"\n')
        f.write(f'output_folder = "{output_folder}"\n')
        f.write(f'model_baylos = "{model_baylos}"\n')
        f.write(f'model_yolov7 = "{model_yolov7}"\n')

# Folder selection functions
def select_folder_images():
    folder = filedialog.askdirectory()
    if folder:
        folder_images_var.set(folder)

def select_output_folder():
    folder = filedialog.askdirectory()
    if folder:
        output_folder_var.set(folder)

def select_model_baylos():
    file = filedialog.askopenfilename(filetypes=[("PTH files", "*.pth")])
    if file:
        model_baylos_var.set(file)

def select_model_yolov7():
    file = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
    if file:
        model_yolov7_var.set(file)

# Function to save the selected paths
def save_paths():
    update_config(folder_images_var.get(), output_folder_var.get(), model_baylos_var.get(), model_yolov7_var.get())
    messagebox.showinfo("Success", "Paths have been updated in config.txt")

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Path Configuration")
root.minsize(400, 200)
root.resizable(True, True)

# Enable grid resizing
root.grid_columnconfigure(1, weight=1)

# Read existing configuration or use default values
(folder_images, output_folder, model_baylos, model_yolov7) = read_config()

# StringVars for each field
folder_images_var = tk.StringVar(value=folder_images)
output_folder_var = tk.StringVar(value=output_folder)
model_baylos_var = tk.StringVar(value=model_baylos)
model_yolov7_var = tk.StringVar(value=model_yolov7)

# Create GUI elements with padding and sticky configuration
tk.Label(root, text="Input Folder:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=folder_images_var).grid(row=0, column=1, padx=10, pady=5, sticky="ew")
tk.Button(root, text="Select", command=select_folder_images).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Output Folder:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=output_folder_var).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
tk.Button(root, text="Select", command=select_output_folder).grid(row=1, column=2, padx=10, pady=5)

tk.Label(root, text="Model Count:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=model_baylos_var).grid(row=2, column=1, padx=10, pady=5, sticky="ew")
tk.Button(root, text="Select", command=select_model_baylos).grid(row=2, column=2, padx=10, pady=5)

tk.Label(root, text="Model Detect:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
tk.Entry(root, textvariable=model_yolov7_var).grid(row=3, column=1, padx=10, pady=5, sticky="ew")
tk.Button(root, text="Select", command=select_model_yolov7).grid(row=3, column=2, padx=10, pady=5)

tk.Button(root, text="Save Paths", command=save_paths).grid(row=4, column=0, columnspan=3, pady=10)

# Start the Tkinter main loop
root.mainloop()
