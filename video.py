import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from utils import non_max_suppression, bbox_iou


cmap = plt.get_cmap('tab20b')
colors = [255*np.asarray(cmap(i)) for i in np.linspace(0, 1, 20)]

def prep_image(image, config):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (config["img_w"], config["img_h"]),
        interpolation=cv2.INTER_LINEAR
    )
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = np.asarray([image])
    return torch.from_numpy(image)

def get_rescaled_coords(ori_h, ori_w, pre_h, pre_w, x1, y1, x2, y2):
    box_h = ((y2 - y1) / pre_h) * ori_h
    box_w = ((x2 - x1) / pre_w) * ori_w
    y1 = (y1 / pre_h) * ori_h
    x1 = (x1 / pre_w) * ori_w
    return x1, y1, box_h, box_w

def main(video_fn):
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python video.py params.py")
        sys.exit()

    params_path = sys.argv[1]

    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()

    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    is_training = False

    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # load pretrained model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            YOLOLoss(
                config["yolo"]["anchors"][i],
                config["yolo"]["classes"],
                (config["img_w"], config["img_h"])
            )
        )

    # load class names
    classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]

    cap = cv2.VideoCapture(video_fn)
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
        if ret == True:
            # 1. pre-process image
            logging.info("processing frame")
            image_tensor = prep_image(frame, config)

            with torch.no_grad():
                outputs = net(image_tensor)
                output_list = []

                for i in range(3):
                    output_list.append(yolo_losses[i](outputs[i]))

                output = torch.cat(output_list, 1)

                batch_detections = non_max_suppression(
                    output, config["yolo"]["classes"],
                    conf_thres=config["confidence_threshold"],
                    nms_thres=0.45
                )

            for idx, detections in enumerate(batch_detections):
                if detections is not None:
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)

                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # Rescale coordinates to original dimensions
                        x1, y1, box_w, box_h = get_rescaled_coords(
                            frame.shape[0], frame.shape[1],
                            config["img_h"], config["img_w"],
                            x1, y1,
                            x2, y2
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x1+box_w, y1+box_h),
                            color, 2
                        )

                        cv2.putText(
                            frame,
                            classes[int(cls_pred)],
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, color,
                            1,
                            cv2.LINE_AA
                        )

            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break
          # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(video_fn='./data/2.mp4')
