import argparse
import sys
import os
import time

import streamlit as st
from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

st.set_page_config(
    page_title="Nhận diện đối tượng 3D KITTI",
    page_icon="🛣️​",
)
print("Current Working Directory:", os.getcwd())
original_directory = '../../../'
new_directory = os.path.abspath(os.path.join(os.getcwd(), 'utility', 'B5_3D_Object_detection_KITTI', 'src'))
print("New Directory Path:", new_directory)
# In ra thư mục làm việc hiện tại
# Kiểm tra xem đường dẫn có tồn tại không
if os.path.isdir(new_directory):
    os.chdir(new_directory)
    print("Changed Working Directory to:", os.getcwd())
else:
    print("Directory does not exist:", new_directory)

# Thêm thư mục chứa config vào sys.path
sys.path.append(new_directory)


# sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format
from data_process.kitti_dataset import KittiDataset  # Import lớp KittiDataset

@st.cache(allow_output_mutation=True)
def load_model(configs):
    model = create_model(configs)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=configs.device))
    model = model.to(device=configs.device)
    model.eval()
    return model

def parse_test_configs(no_cuda, pretrained_path):
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=pretrained_path, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true', default=no_cuda,
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demonstration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    args = parser.parse_args(args=[])
    configs = edict(vars(args))
    configs.pin_memory = True

    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    configs.device = torch.device('cuda' if torch.cuda.is_available() and not configs.no_cuda else 'cpu')
    
    return configs

def main():
    st.title("3D Object Detection On KITTI Dataset")
    st.write("Upload an image to get predictions from the YOLOv4 model.")

    no_cuda = st.checkbox('Disable CUDA (use CPU only)', value=True)
    pretrained_path = st.text_input('Path to pretrained model checkpoint', value='..\model\complex_yolov4_mse_loss.pth')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    configs = parse_test_configs(no_cuda, pretrained_path)
    model = create_model(configs)

    device_string = 'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx)
    
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=device_string))

    configs.device = torch.device(device_string)
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()
    if uploaded_file is not None:
        index, _ = os.path.splitext(uploaded_file.name)
        index = int(index)
        # Tạo đối tượng KittiDataset và xử lý ảnh
        dataset = KittiDataset(configs.dataset_dir, mode='test')
        img_paths, imgs_bev = dataset.load_img_only(index)
        print('img_paths ', img_paths)
        print('imgs_bev ', imgs_bev)
        st.image(img_paths, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
        st.write("Running inference...")
        imgs_bev = torch.from_numpy(imgs_bev)
        input_imgs = imgs_bev.to(device=configs.device).float().unsqueeze(0)
        t1 = time_synchronized()
        outputs = model(input_imgs)
        t2 = time_synchronized()
        detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        img_bev = imgs_bev.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
        for detections in img_detections:
            if detections is None:
                continue
            # Rescale boxes to original image
            detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
            for x, y, w, l, im, re, *_, cls_pred in detections:
                yaw = np.arctan2(im, re)
                label = cnf.class_list[int(cls_pred)]
                # Draw rotated box
                kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)], label)
                    
        img_rgb = cv2.imread(img_paths)
        calib = kitti_data_utils.Calibration(img_paths.replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
        img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption= 'Imge from camera with 3D object detection', use_column_width=True)

        img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)
        img_bev = cv2.cvtColor(img_bev, cv2.COLOR_BGR2RGB)
        
        st.image(img_bev, caption= 'Imge from Lidar with 2D object detection', use_column_width=True)
        st.write(f"Done processing. Time: {t2 - t1:.1f}ms, Speed: {1 / (t2 - t1):.2f} FPS")
    
    os.chdir(original_directory)
    print('current_directory: ', os.getcwd())
    
if __name__ == '__main__':
    main()
    

