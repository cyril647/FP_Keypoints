import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from skimage import io
from matplotlib import pyplot as plt

###### plot exemplary images from COCO
# import json
# with open('/home/fischer/ImageDatasets/coco/annotations/person_keypoints_minival2014.json') as f:
# 	test = json.load(f)
# for im in test['images']:
# 	plt.imshow(io.imread(os.path.join('/home/fischer/ImageDatasets/coco/val2014/', im['file_name'])))
# 	plt.show()

coco = COCO('/home/fischer/ImageDatasets/coco/annotations/person_keypoints_minival2014.json')
catIds = coco.getCatIds(catNms=['person'])  # catIds=1 means people
keys = list(coco.imgs.keys())
keys.sort()
for key in keys:
	im = coco.imgs[key]
	# im['id'] = 328
	img = coco.loadImgs(im['id'])[0]
	image_path = os.path.join('/home/fischer/ImageDatasets/coco/val2014/', 'COCO_val2014_' + str(im['id']).zfill(12) + '.jpg')
	I = io.imread(image_path)
	plt.axis('off')
	plt.imshow(I)  # Draw the image and show it to plt.show()
	bg = np.zeros((img['height'], img['width'], 3))
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
	anns = coco.loadAnns(annIds)
	coco.showAnns(anns)
	plt.title(str(im['id']))
	plt.show()  # Display image
#########################################

### plot for DIV2K keypoints
# from https://www.programmersought.com/article/30024173559/
# dataset_dir = 'datasets/DIV2K_keypoints'
#
# coco = COCO(os.path.join(dataset_dir, 'labels', 'test_keypoint_with_crowd_converted.json'))
# catIds = coco.getCatIds(catNms=['person'])  # catIds=1 means people
# imgIds = 434
# img = coco.loadImgs(imgIds)[0]
# image_path = os.path.join(dataset_dir, 'DIV2K_images', str(imgIds-400).zfill(4) + '.png')
# I = io.imread(image_path)
# plt.axis('off')
# plt.imshow(I)  # Draw the image and show it to plt.show()
# bg = np.zeros((img['height'], img['width'], 3))
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
# plt.show()  # Display image