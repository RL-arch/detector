# sourcery skip: merge-list-append, merge-list-appends-into-extend, merge-list-extend

from organize.preprocess import rename, preprocess
from organize.framediff import difference
from utils.count import docount

extension = ".tif"
image_size = 512


#-----------------------------------------------------------------------------------
# ! Modify the paths and file name prefixs here
# folder with all the images
folder_images = "/Users/bunkyotop/Documents/Iamges/Input"
# path for the model to count organoids
model_baylos = "/Users/bunkyotop/Library/CloudStorage/OneDrive-KULeuven/code/CF/CFAnalyzer/trained_models/bayersian/best_model.pth"
# where you want to store the results
output_folder = "/Users/bunkyotop/Documents/Iamges/Output"

prefix = []
# prefix.append("242-CF PE FIS1 20210705 timelapse-01")
# prefix.append("Exp 87 242-CF PE FIS2 20211007 tl 2h-01")
# prefix.append("202112089 MB exp100 242-CF PE time series2-01")
# prefix.append("20220215 MB exp111 242-CF PE timelapse-01")
# prefix.append("20220506 MB 242-CF PE timelapse 2h-01-Image Export-01")
# prefix.append("242-CF PE FIS 20220511 exp129-try2 time-01")
prefix.append("429-CF en 242-CF PE FIS 20220602 tl-01")
# prefix.append("429-CF en 242-CF PE FIS 20220630 time series-01")
# prefix.append("exp146 426-CF en 242-CF PE FIS 20220706 timelapse-01")
# prefix.append("exp147 439-CF and 079-CF L227R-N1303K Hz n1 20220719 timeseries-01")
# prefix.append("exp147 426-CF and 242-CF L227R-N1303K Ho c1-12 20220719 timeseries-01")
# prefix.append("exp 148 439-CF en 79-CF PE FIS 20220727 time laps-01")
# prefix.append("exp 148 426-CF en 242-CF PE FIS 20220727 time laps-01")
# prefix.append("exp151 426-CF L227R 20220902 timeseries-01")
# prefix.append("exp153 426-CF L227R 20220915 time series-01")
# prefix.append("exp155 242en435-CF N1303K 20220915timeseries-01")
# prefix.append("exp159 242en435-CF N1303K 20220921timeseries-01")
# unique string sequence for exp
#-----------------------------------------------------------------------------------

if __name__ == '__main__':
    num_exps = rename(folder_images)
    start_folder, end_folder = preprocess(folder_images, output_folder,
                                          num_exps)
    difference(start_folder, end_folder, output_folder, extension)
    docount(start_folder, model_baylos, output_folder)
