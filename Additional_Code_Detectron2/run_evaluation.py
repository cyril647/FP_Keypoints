import os
import sys
import argparse
import shutil

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg

from detectron2.data.datasets.builtin import register_all_cityscapes
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor

import torch

# CONFIDENCE_THRESHOLD = 0.3

RESULT_FILE_BASE = 'resultInstanceLevelSemanticLabeling'


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
    if not args.inputPath or not args.outputPathClassifiedImages or not args.outputPathJson or not args.outputSuffix:
        logger.error('Not enough input arguments given!')
        return 1
        
    cfg = setup_cfg(args)
    ## possible cityscapes options:
    # cityscapes_fine_instance_seg_train
    # cityscapes_fine_sem_seg_train
    # cityscapes_fine_instance_seg_val
    # cityscapes_fine_sem_seg_val
    # cityscapes_fine_instance_seg_test
    # cityscapes_fine_sem_seg_test
    
    # delte content of output folder and create new afterwards
    if not os.path.isdir(args.outputPathClassifiedImages):
        os.makedirs(args.outputPathClassifiedImages)
        # shutil.rmtree(args.outputPathClassifiedImages)
        
    if not os.path.isdir(args.outputPathJson):
        os.makedirs(args.outputPathJson)
    
    resultJsonFile = os.path.join(args.outputPathJson, RESULT_FILE_BASE + args.outputSuffix + '.json')
    
    # continue if output json file already exists
    if os.path.isfile(resultJsonFile):
        logger.warn(resultJsonFile + ' exists already!')
        return 2
        
    cityscapesSplits = {
        "cityscapes_fine_{task}_train": (os.path.join(args.inputPath, "train"), "cityscapes/gtFine/train"),
        "cityscapes_fine_{task}_val": (os.path.join(args.inputPath, "val"), "cityscapes/gtFine/val"),
        "cityscapes_fine_{task}_test": (os.path.join(args.inputPath, "test"), "cityscapes/gtFine/test"),
    }
    
    model = []
    if 'mask' in args.outputPathJson:
        model = 'mask'
    elif 'faster' in args.outputPathJson:
        model = 'faster'
    else:
        logger.error('Unknown model')
        return 1
    
    register_all_cityscapes(leaveMeta=True, cityscapesSplits=cityscapesSplits, model=model)
    
    data_loader = build_detection_test_loader(cfg, "cityscapes_fine_instance_seg_val")  # "custom_dataset")
    
    predictor = DefaultPredictor(cfg)
    
    evaluator = CityscapesEvaluator(dataset_name="cityscapes_fine_instance_seg_val",
                                    outputPath=args.outputPathClassifiedImages,
                                    jsonOutput=resultJsonFile)  # , #"custom_dataset",
    
    inference_on_dataset(predictor.model, data_loader, evaluator)
    
    print("Done.")


if __name__ == "__main__":
    args = get_parser().parse_args()
    ret = run_evaluation(args)
    sys.exit(ret)
