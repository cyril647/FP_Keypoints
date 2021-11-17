#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
get_ipython().system('gcc --version')


# In[2]:


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[3]:



cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)
# outputs = predictor(im)


# In[4]:


# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
register_coco_instances("test_keypoint", {}, "/HOMES/sichengliu/Train_Image/test_keypoint.json", "/HOMES/sichengliu/Train_Image/")


# In[5]:


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("test_keypoint", ("bbox", "keypoints"), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "test_keypoint")
# print(val_loader)
# print(evaluator)
print(inference_on_dataset(predictor.model, val_loader, evaluator))

