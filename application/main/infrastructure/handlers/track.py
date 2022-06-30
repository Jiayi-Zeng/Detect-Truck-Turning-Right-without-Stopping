# limit the number of cpus used by high performance libraries

import os
import warnings
import platform
import shutil
from pathlib import Path
import cv2
import copy
import numpy as np
import time
# import dlib
import os
import subprocess


import torch
import torch.backends.cudnn as cudnn
import sys
from infrastructure.yolov5.models.experimental import attempt_load
from infrastructure.yolov5.utils.downloads import attempt_download
from infrastructure.yolov5.models.common import DetectMultiBackend
from infrastructure.yolov5.utils.datasets import LoadImages, LoadStreams
from infrastructure.yolov5.utils.general import LOGGER, check_img_size, increment_path, non_max_suppression, \
    scale_coords, check_imshow, xyxy2xywh, increment_path
from infrastructure.yolov5.utils.torch_utils import select_device, time_sync
from infrastructure.yolov5.utils.plots import Annotator, colors

from infrastructure.deep_sort_pytorch.utils.parser import get_config
from infrastructure.deep_sort_pytorch.deep_sort import DeepSort
import argparse

from util.common import read_yml, extract_xywh_hog
from util.OPT_config import OPT

from infrastructure.helper.zone_drawer_helper import ZoneDrawerHelper

from infrastructure.database.Vehicle import Vehicle
from infrastructure.database.common import add_vehicle_to_db

from threading import Thread
from datetime import timedelta, datetime
import math

# environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# sys.path.insert(0, './yolov5')
lib_path = os.path.abspath(os.path.join('infrastructure', 'yolov5'))
sys.path.append(lib_path)


class Tracker:
    def __init__(self, config_path: str) -> None:
        config = read_yml(config_path)
        self.opt = OPT(config=config)
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand

    def detect(self):

        # Initialize
        opt = self.opt
        out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, save_csv, imgsz, evaluate, half, \
        upper_ratio, right_ratio, = opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, \
                                    opt.save_vid, opt.save_txt, opt.save_csv, opt.imgsz, opt.evaluate, opt.half, \
                                    opt.upper_ratio, opt.right_ratio
        zone_drawer = ZoneDrawerHelper()
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        device = select_device(opt.device)
        half &= device.type != 'cpu'  # half precision only on CUDA
        M, M_inverse = cal_perspective_params()

        # Initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        if not evaluate:
            if os.path.exists(out):
                pass
                # shutil.rmtree(out)  # delete output folder
            else:
                os.makedirs(out)  # make new output folder

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(opt.yolo_weights, device=device, dnn=opt.dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half &= pt and device.type != 'cpu'  # half precision only suvehiclesorted by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()

        # display or not
        if show_vid:
            show_vid = check_imshow()

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
            total_time = get_video_duration(source)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        save_path = str(Path(out))
        # extract what is in between the last '/' and last '.'
        txt_file_name = source.split('/')[-1].split('.')[0]
        txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
        csv_path = str(Path(out)) + '/' + txt_file_name + '.csv'

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        previous_frame, current_frame = [-1, -1]
        vehicle_infos = {}  # id : { start in view, exit view, type}
        # LIST CONTAIN vehicles HAS APPEARED, IF THAT VEHICLE HAD BEEN UPLOADED TO DB, REMOVE THAT VEHICLE
        list_vehicles = set()

        # detect and track
        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            frame_height = im0s.shape[0]
            frame_width = im0s.shape[1]
            upper_line = int(frame_height * upper_ratio)
            right_line = int(frame_width * right_ratio)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference - YOLOv5
            visualize = increment_path(save_path / Path(path).stem, mkdir=True) if opt.visualize else False
            pred = model(img, augment=opt.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Avehiclesly NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                       max_det=opt.max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    # s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                save_path = str(Path(out) / Path(p).name)

                annotator = Annotator(im0, line_width=2, pil=not ascii)

                # draw red zones
                zone_drawer.draw(im0, frame_width=frame_width, frame_height=frame_height, upper_ratio=upper_ratio,
                                 right_ratio=right_ratio)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort, only objects in used zone
                    xywhs = np.asarray(xywhs.cpu())
                    confs = np.asarray(confs.cpu())
                    clss = np.asarray(clss.cpu())

                    # Exclude area of unused targets
                    row_indexes_delete = []
                    for index, cord in enumerate(xywhs):
                        if not (cord[1] > upper_line and cord[0] > right_line):
                            row_indexes_delete.append(index)
                    xywhs = np.delete(xywhs, row_indexes_delete, axis=0)
                    confs = np.delete(confs, row_indexes_delete)
                    clss = np.delete(clss, row_indexes_delete)

                    # deepsort
                    xywhs = torch.tensor(xywhs)
                    confs = torch.tensor(confs)
                    clss = torch.tensor(clss)
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    current_frame = {'time': total_time * frame/dataset.frames,
                                     'frame': frame_idx,
                                     'n_vehicles_at_time': len(outputs),
                                     'IDs_vehicles': []}

                    if len(outputs) > 0:
                        current_frame['IDs_vehicles'] = list(outputs[:, 4])

                    if (current_frame != -1) and (previous_frame != -1):
                        previous_IDs = previous_frame['IDs_vehicles']
                        current_IDs = current_frame['IDs_vehicles']

                        # for ID in current_IDs:
                        for ID in current_IDs:
                            #  for ID not in previous_IDs:
                            if (ID not in previous_IDs) and (ID not in list_vehicles):
                                vehicle_infos[ID] = {}
                                vehicle_infos[ID]['in_time'] = current_frame['time']
                                vehicle_infos[ID]['exit_time'] = float('inf')
                                vehicle_infos[ID]['type_vehicle'] = 'vehicle'
                                vehicle_infos[ID]['temporarily_disappear'] = 0
                                vehicle_infos[ID]['route'] = {}
                                vehicle_infos[ID]['new_route'] = {}
                                vehicle_infos[ID]['speed'] = {}

                        # for ID in previous_IDs:
                        for ID in copy.deepcopy(list_vehicles):
                            # for ID not in current_IDs:
                            if ID not in current_IDs:
                                vehicle_infos[ID]['temporarily_disappear'] += 1
                                # 25 frame ~ 1 seconds
                                if (vehicle_infos[ID]['temporarily_disappear'] > 200) and \
                                        (current_frame - vehicle_infos[ID]['in_time']) > timedelta(seconds=3):
                                    vehicle_infos[ID]['exit_time'] = current_frame
                                    str_ID = str(ID) + "-" + str(time.time()).replace(".", "")
                                    if opt.upload_db:
                                        this_vehicle = Vehicle(str_ID, vehicle_infos[ID]['in_time'],
                                                               vehicle_infos[ID]['exit_time'],
                                                               vehicle_infos[ID]['type_vehicle'])
                                        Thread(target=add_vehicle_to_db, args=[this_vehicle]).start()

                                    list_vehicles.discard(ID)

                    # Cal info and Visualize deep-sort outputs
                    # info
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            # base info
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            c = int(cls)  # integer class
                            label = f'{names[c]}- id {id}'
                            vehicle_infos[id]['type_vehicle'] = names[c]

                            # add route
                            bbox_left, bbox_top, bbox_right, bbox_bottom = bboxes
                            center = (bbox_left + bbox_right) // 2, (bbox_top + bbox_bottom) // 2
                            vehicle_infos[id]['route'][current_frame['time']] = center

                            # add new_route
                            Z = M[2, 0] * center[0] + M[2, 1] * center[1] + M[2, 2]
                            tempX = (M[0, 0] * center[0] + M[0, 1] * center[1] + M[0, 2]) // Z
                            tempY = (M[1, 0] * center[0] + M[1, 1] * center[1] + M[1, 2]) // Z
                            new_point = [int(tempX), int(tempY)]
                            vehicle_infos[id]['new_route'][current_frame['time']] = new_point

                            # cal speed
                            last = []
                            for now in vehicle_infos[id]['new_route'].items():
                                if not last == []:
                                    now_time, now_point = now[0], now[1]
                                    last_time, last_point = last[0], last[1]
                                    del_time = now_time - last_time
                                    del_x = now_point[0] - last_point[0]
                                    del_y = now_point[1] - last_point[1]
                                    dis = math.sqrt(del_x ** 2 + del_y ** 2)
                                    speed = int((3.5 * dis / 160) / (del_time * 0.001) * 3.6)
                                    vehicle_infos[id]['speed'][now_time] = speed
                                last = now

                            # print route
                            if current_frame["frame"] > 2:
                                for index, value in vehicle_infos.items():
                                    if current_frame['time'] < value['exit_time']:
                                        route = value["route"]

                                        # draw route
                                        last = []
                                        for now in route.items():
                                            if not last == []:
                                                now_time, now_point = now[0], now[1]
                                                last_time, last_point = last[0], last[1]
                                                # draw
                                                cv2.line(im0, now_point, last_point, (255, 0, 0), thickness=2,
                                                         lineType=8)
                                            last = now

                                        # draw speed
                                        ns = 5
                                        print(len(vehicle_infos[id]['speed']))
                                        if len(vehicle_infos[id]['speed']) % ns == 0 and len(vehicle_infos[id]['speed']) != 0:
                                            avr_speed = 0
                                            for time, speed in vehicle_infos[id]['speed']:
                                                avr_speed += speed
                                            vehicle_infos[id]['speed'] = {}
                                            avr_speed //= ns
                                            vehicle_infos[id]['speed'] = {}
                                            label = f'{names[c]}- id {id} {avr_speed}km/h'

                            # print box
                            annotator.box_label(bboxes, label, color=colors(c, True))

                        vehicles_count = current_frame['n_vehicles_at_time']
                        IDs_vehicles = current_frame['IDs_vehicles']
                        LOGGER.info("{}: {} vehicles".format(s, vehicles_count))

                        if not np.isnan(np.sum(IDs_vehicles)):
                            list_vehicles.update(list(IDs_vehicles))
                else:
                    deepsort.increment_ages()

                # Stream results
                im0 = annotator.result()
                if show_vid:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            previous_frame = current_frame

        # Print results
        print(vehicle_infos)
        print(list_vehicles)
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_vid or save_csv:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)


def cal_perspective_params():
    points = [[359, 307], [461, 306], [312, 542], [497, 542]]
    src = np.float32(points)
    # 俯视图中四点的位置
    dst = np.float32([[359, 400], [461, 400], [359, 542], [461, 542]])
    # 从原始图像转换为俯视图的透视变换的参数矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 从俯视图转换为原始图像的透视变换参数矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    return M, M_inverse


def get_video_duration(video_path: str):
    ext = os.path.splitext(video_path)[-1]
    if ext != '.mp4' and ext != '.avi' and ext != '.flv':
        raise Exception('format not support')
    ffprobe_cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'
    p = subprocess.Popen(
        ffprobe_cmd.format(video_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    out, err = p.communicate()
    duration_info = float(str(out, 'utf-8').strip())
    return int(duration_info * 1000)

if __name__ == '__main__':
    tracker = Tracker(config_path='../settings/config.yml')

    with torch.no_grad():
        tracker.detect()
