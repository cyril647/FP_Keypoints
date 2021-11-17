import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt

def main():
	im = cv2.imread('datasets/DIV2K_keypoints/DIV2K_images/0008.png')
	im = cv2.imread('datasets/DIV2K_keypoints/DIV2K_images/0181.png')
	cfg = get_cfg()
	# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
	cfg.merge_from_file('configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')
	cfg.MODEL.WEIGHTS = 'models/COCO/keypoint_rcnn_R_50_FPN_x3/model_final_a6e10b.pkl'
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
	predictor = DefaultPredictor(cfg)
	outputs = predictor(im)
	v1 = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
	out1 = v1.draw_instance_predictions(outputs["instances"].to("cpu"))
	eg1 = out1.get_image()[:, :, ::-1]
	plt.figure(figsize=(20, 20))
	plt.axis("off")
	img1 = cv2.cvtColor(eg1,cv2.COLOR_BGR2RGB)
	plt.imshow(img1)
	plt.show()


if __name__ == '__main__':
    main()
