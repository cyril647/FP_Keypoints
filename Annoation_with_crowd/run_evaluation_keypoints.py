import os
import sys
import argparse
import shutil

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg

from detectron2.data.datasets.builtin import register_all_cityscapes
from detectron2.data.datasets.builtin import register_all_coco
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from tests.data.test_coco_evaluation import TestCOCOeval

from detectron2.data.datasets.builtin_meta import KEYPOINT_CONNECTION_RULES, COCO_PERSON_KEYPOINT_FLIP_MAP, COCO_PERSON_KEYPOINT_NAMES

import torch

# CONFIDENCE_THRESHOLD = 0.3

RESULT_FILE_BASE = 'resultKeypointDetection'


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # put model on gpu
    cfg.MODEL.DEVICE = "cuda:0"
    cfg.merge_from_file(args.configFile)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidenceThreshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidenceThreshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidenceThreshold
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATASETS.TEST = ("cityscapes_fine_instance_seg_val",)
    cfg.DATALOADER.NUM_WORKERS = 0  # has to be set to zero for debugging c.f. https://superuser.com/questions/1473687/pycharm-debugger-does-not-work-with-pytorch-and-deep-learning
    cfg.freeze()
    
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument(
        "--configFile",
        default='',
        # default="../configs/Cityscapes/mask_rcnn_R_50_FPN.yaml",
        # default="../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    
    parser.add_argument(
        "--outputPathJson",
        # default='datasets/cityscapes/evaluationResults/',
        help="A directory to save JSON outputs from evaluation. "
    )
    
    parser.add_argument("--outputPathClassifiedImages",
                        # default='datasets/cityscapes/results/',
                        help="A file or directory to save images with classification results. "
                        )
    
    parser.add_argument("--inputPath",
                        # default='/CLUSTERHOMES/fischer/cityscapes_vtm/leftImg8bitCompressed/',
                        help="Directory where input images to classify are saved. "
                        )

    parser.add_argument("--gtInstancesPath",
                        help="Directory where PNGs with gt instances are saved. "
                        )
    
    parser.add_argument("--outputSuffix",
                        # default='/CLUSTERHOMES/fischer/cityscapes_vtm/leftImg8bitCompressed/',
                        help="Suffix for output files. E.g. _qp_32_filter_0_0_0. "
                        )
    
    parser.add_argument(
        "--confidenceThreshold",
        type=float,
        # default=CONFIDENCE_THRESHOLD,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


# main script to call to run inference of selected model on selected dataset
def run_evaluation(args):

    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     y = torch.rand(1, 1, device=device)
    #
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    if not args.inputPath or not args.outputPathJson or not args.outputSuffix:
        logger.error('Not enough input arguments given!')
        return 1
    
    # create output folder
    if not os.path.isdir(args.outputPathClassifiedImages) and args.outputPathClassifiedImages:
        os.makedirs(args.outputPathClassifiedImages, exist_ok=True)

    if not os.path.isdir(args.outputPathJson):
        os.makedirs(args.outputPathJson, exist_ok=True)

        
    ##### here is only for Mask and Faster R-CNN and no YOLO ######
    cfg = setup_cfg(args)
    ## possible cityscapes options:
    # cityscapes_fine_instance_seg_train
    # cityscapes_fine_sem_seg_train
    # cityscapes_fine_instance_seg_val
    # cityscapes_fine_sem_seg_val
    # cityscapes_fine_instance_seg_test
    # cityscapes_fine_sem_seg_test
    
    resultJsonFile = os.path.join(args.outputPathJson, RESULT_FILE_BASE + args.outputSuffix + '.txt')
    
    # continue if output json file already exists
    if os.path.isfile(resultJsonFile):
        if os.path.getsize(resultJsonFile) > 0:
            logger.warning(resultJsonFile + ' exists already!')
            return 2

    addMeta = {'keypoint_connection_rules' : KEYPOINT_CONNECTION_RULES,
               'keypoint_flip_map' : COCO_PERSON_KEYPOINT_FLIP_MAP,
               'keypoint_names' : COCO_PERSON_KEYPOINT_NAMES}
    register_coco_instances('keypoint_DIV2K', addMeta, args.gtInstancesPath, args.inputPath)

    data_loader = build_detection_test_loader(cfg, 'keypoint_DIV2K')  # "custom_dataset")

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator('keypoint_DIV2K', cfg, True, output_dir=args.outputPathJson, output_file_name=resultJsonFile, outputFilePath=args.outputPathClassifiedImages)
    evaluator.reset()
    
    inference_on_dataset(predictor.model, data_loader, evaluator)
    
    print("Done.")
    

if __name__ == "__main__":
    args = get_parser().parse_args()
    ret = run_evaluation(args)
    sys.exit(ret)
