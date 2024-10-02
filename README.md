# DETECTOR: AI-based Organoid Detection for Automated FIS Analysis

Implementation the software in paper - **[Prime editing functionally corrects Cystic Fibrosis-causing CFTR mutations in human organoids and airway epithelial cells](https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(24)00234-9)** (Cell Reports Medicine)

This code serves two purposes: firstly, it accurately counts the total number of organoids in each image using Bayesian Crowd Counting, resolving the issue of segmentation inaccuracy in dense organoids. Secondly, it identifies swelling organoids resulting from gene editing using YOLOv7.

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{bulcaenPrimeEditingFunctionally2024,
  title = {Prime Editing Functionally Corrects Cystic Fibrosis-Causing {{CFTR}} Mutations in Human Organoids and Airway Epithelial Cells},
  author = {Bulcaen, Mattijs and Kortleven, Phéline and Liu, Ronald B. and Maule, Giulia and Dreano, Elise and Kelly, Mairead and Ensinck, Marjolein M. and Thierie, Sam and Smits, Maxime and Ciciani, Matteo and Hatton, Aurelie and Chevalier, Benoit and Ramalho, Anabela S. and family=Solvas, given=Xavier Casadevall, prefix=i, useprefix=false and Debyser, Zeger and Vermeulen, François and Gijsbers, Rik and Sermet-Gaudelus, Isabelle and Cereseto, Anna and Carlon, Marianne S.},
  date = {2024-05-01},
  journaltitle = {Cell Reports Medicine},
  issn = {2666-3791},
  doi = {10.1016/j.xcrm.2024.101544},
  url = {https://www.cell.com/cell-reports-medicine/abstract/S2666-3791(24)00234-9}
}
```

## Overview

<p align="center">
<img src=20240422_Detector_GitHub-scheme-1.png>
</p>

**a)** Organoids swelling with CFTR gene editing;
**b)** Bayesian Crowd Counting for dense organoids & Swelling detection;
**c)** Swelling detection results

## Datasets and Trained models

[Images for detection](https://)

[Datasets for training](https://drive.google.com/drive/folders/1LzXNgAhEYSCjYba_eXknL6RFqJa7Zac1?usp=drive_link)

Examples of the images used for testing can be found in the [data](./data/Input/) folder.

Examples of the results can be found in the [data/Output](./data/Output/) folder.

Trained models please download from [Google Drive](https://drive.google.com/drive/folders/1jLQce54EvSB6dyfhKNpMT_4n9YugL6sl?usp=sharing). The folder organization is same as the [data/trained_models](./data/trained_models/) folder.

## Installation and Run
### 1. Environment Setup
****************

There are two ways to set up your environment:

**Option1**

`conda env create -f environment.yml`

**Option2 (Recommended)**

create an Anaconda environment with the name you want:

`conda create --name <your name>`

then, install the packages required:

`pip install -r requirements.txt`

as we tested with Intel Macbook Pro 2020, xlsxwriter also needed to be installed:

`conda install -c conda-forge xlsxwriter`

**Option3**

use Docker:

``` bash
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov7 -it -v your_coco_path/:/coco/ -v your_code_path/:/yolov7 --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop


```

>

### 2. Prepare the data you want to analyze
***************************************************
2.1 Organize your image folder

Image sequences from each experiment should be collected into one folder. These images, should have the same name format. 


for example: 

**exp1 242en435-CF N1303K 20220921timeseries-01**_s01t01.tif

**exp1 242en435-CF N1303K 20220921timeseries-01**_s01t02.tif

**exp1 242en435-CF N1303K 20220921timeseries-01**_s01t03.tif

**exp1 242en435-CF N1303K 20220921timeseries-01**_s01t04.tif

....

**exp1 242en435-CF N1303K 20220921timeseries-01**_s01t013.tif

**exp1 242en435-CF N1303K 20220921timeseries-01**_s02t01.tif

**exp1 242en435-CF N1303K 20220921timeseries-01**_s02t02.tif

....

**exp1 242en435-CF N1303K 20220921timeseries-01**_s02t013.tif

....

And for this experiment, "
**exp1 242en435-CF N1303K 20220921timeseries-01** " will be the _"prefix"_

Then, put these experiment data folders into **one folder**,
for example: "/Input"

In the _/Input_ there are 3 folders: _exp1, exp 2, exp 3_

    Image
        |
        |__exp1
        |
        |__exp2
        |
        |__exp3

>

2.2 Organize your image filenames

~~The test image names are following the format from ZEISS software, and with the experiment time of 2h; ~~
If your experiment time is 1h, please use `/data/rename.py` to rename the files.

Therefore, the names are with suffix _s...t01_ to _s...t13_

**!!Important**
The dynamic morphological changes are based on the detection of dynamic morphological changes from the **t02** to **t12**.

If your experiment is 2h: the image sequences are t01 t02 t03 t04 t05 t06 t07 t08 t09 t10 t11 t12 (t13)

If your experiment is 1h: the image sequences are t00 t02 t04 t06 t08 t10 t12

>

********************************

### 3. Modify the codes
### (Operations under _/detector_ folder)

****************

* In the [bayesian/detect_1.py](./detector/bayesian/detect_1.py#L11-L18), modify the paths of your _Image_ folder descirbed in **2.1** (**line 11~38**, indicated in the comments):

  `folder_images = <your image path>`

    Load the trained model to count the total number, we put the model in the directory _/trained_models/bayesian/best_model.pth_. Copy and paste the location of this file into the path:

    `model_baylos = <the path of trained VGG19 model>`

    _Output_ folder you want:

    `output_folder = <out path your want>`


* In the [yolov7/detect_2.py](./detector/yolov7/detect_2.py#L377-L383) (**line 377~383**, indicated in the comments):, copy and paste the paths of your _Output_ folder:

  `output_folder = <same as in detect_1.py>`

    load the trained model to count the swelling organoids, which is a yolov7 model, we put the model in the directory _/trained_models/yolov7/last.pt_. Copy and paste the location of this file into the path:

    `model_yolov7 = <the path of trained yolov7 model>`

Now, the modifications are completed.  

>


### 4. Run the codes
### (Operations under _/detector_ folder)

*****************

**Option1** 
### Run script directly 

`sh run.sh`

**Option2**
### Manually run on terminal:

 open the terminal in this folder and

`cd bayesian`

Then, run the code:

`python detect_1.py`

Then go to the directory _/yolov7_

`cd .. && cd yolov7`

Run the code:

`python detect_2.py`

The indications are shown on the terminal.

>
### Update notes:
*****************
- (**2024**) we use `writer.close()` to replace `writer.save()` with new version of Pandas version >=1.2.0. If you encounter any problems, please reinstall pandas refer to the [Pandas document](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).

- (**2024**) prefix inputs are not required in the new version of Pandas.

### Known issues

*****************

### 1 Data quality

The brightness of the microscopic image will influence the results of total number estimation. When the image condition is dark and dense crowd, the total number estimate can be less than actual.

The position shift will influence the swelling organoids detection and will make fewer organoids detected. 

~~### 2 Fonts~~

~~Fonts in different OS have different routes and may need to modify in `bayesian/utils/count.py `  (line 52~55)~~

### 3 Internet connection

The network needs to stay on and be able to connect to Google to download initial files like model weights.



## Acknowledgements and model re-training guidance:

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/ZhihengCV/Bayesian-Crowd-Counting](https://github.com/ZhihengCV/Bayesian-Crowd-Counting)

>
