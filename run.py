import path_config as config

from organize.final import finalize
from organize.preprocess import rename, preprocess, resize_images
from organize.framediff import difference
from utils.count import do_count
from engine import do_detect

import time

if __name__ == '__main__':
    '''
    our default set for the raw images is 512x512
    default confidence threshold is 0.3
    '''
    img_size = 512
    conf_thred = 0.3
    
    # Use config variables
    folder_images = config.folder_images
    output_folder = config.output_folder
    model_baylos = config.model_baylos
    model_yolov7 = config.model_yolov7

    # start timer
    start_time = time.time()

    resize_images(folder_images, img_size)
    num_exps = rename(folder_images)
    start_folder, end_folder = preprocess(folder_images, output_folder, num_exps)
    difference(start_folder, end_folder, output_folder)
    do_count(start_folder, model_baylos, output_folder)
    diff_images = f'{output_folder}/diff_images'
    do_detect(model_yolov7, diff_images, img_size, conf_thred, output_folder)
    finalize(output_folder)

    # end timer
    end_time = time.time()
    print(f"\033[1mTotal time taken: {(end_time - start_time)/60:.3f} minutes\033[0m")