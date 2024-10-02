import argparse
import time
from pathlib import Path
import os
import pandas as pd

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import diff, random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from matplotlib import pyplot as plt
from PIL import Image
import shutil


def detect(opt):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith(
        '.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name,
                       exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(
            torch.load('weights/resnet101.pt',
                       map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    obj_list = []
    if device.type != 'cpu':
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1,
                                          0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    obj_dict = {'Name': p.name, 'Swelling': int(n)}
                    obj_list.append(obj_dict.copy())

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                                gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (
                            cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        label = f'{conf:.2f}'
                        plot_one_box(xyxy,
                                     im0,
                                     label=label,
                                     color=(144, 238, 144),
                                     line_thickness=1)
                        im0 = cv2.putText(im0,
                                          f"Est. amount of swelling organoids: {n}",
                                          (1, 20),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7,
                                          color=(255, 255, 255),
                                          thickness=2)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(
                        f" The image with the result is saved in: {save_path}")
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release(
                            )  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    return obj_list


# make dirs with mode
def mkdir_with_mode(directory, mode):
    if not os.path.isdir(directory):
        oldmask = os.umask(000)
        os.makedirs(directory, 0o777)
        os.umask(oldmask)


def do_detect(weights_i, sources_i, img_size_i, conf_thres_i, output_dir_i):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default=weights_i,
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=sources_i,
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size',
                        type=int,
                        default=img_size_i,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=conf_thres_i,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--device',
                        default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf',
                        action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave',
                        action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes',
                        nargs='+',
                        type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument('--update',
                        action='store_true',
                        help='update all models')
    parser.add_argument('--project',
                        default=output_dir_i,
                        help='save results to project/name')
    parser.add_argument('--name',
                        default='img3',
                        help='save results to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace',
                        action='store_true',
                        help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['pt']:
                yolo_list = detect(opt)
                strip_optimizer(opt.weights)
        else:
            yolo_list = detect(opt)
    # return yolo_list
    writer = pd.ExcelWriter(f"{output_dir_i}/excel/swelling amount.xlsx",
                            engine='xlsxwriter')

    df = pd.DataFrame.from_dict(yolo_list)
    df.to_excel(writer, index=False, header=True)
    # writer.save()
    writer.close()
    print("Excel saved in /excel.")
    print(yolo_list)


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
    print(f"cached in {output_dir_o} released.")

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


if __name__ == '__main__':
    img_size = 512
    conf_thred = 0.3
    # ! modify the paths here:
    # --------------------------------------------------------------------------------------
    model_yolov7 = '../../data/trained_models/yolov7/last.pt'
    output_folder = '../../data/Output'
    # --------------------------------------------------------------------------------------
    diff_images = f'{output_folder}/diff_images'
    do_detect(model_yolov7, diff_images, img_size, conf_thred, output_folder)
    finalize(output_folder)
