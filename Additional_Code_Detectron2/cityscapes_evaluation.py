# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import glob
import logging
import os
import tempfile
from collections import OrderedDict
import torch
from PIL import Image
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .evaluator import DatasetEvaluator


class CityscapesEvaluator(DatasetEvaluator):
    """
    Evaluate instance segmentation results using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
    """

    def __init__(self, dataset_name, outputPath='', jsonOutput='output_evaluation.json'):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        if outputPath == '':
            self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
            self._temp_dir = self._working_dir.name
        else:
            self._temp_dir = outputPath
        self._output_json_file = jsonOutput
        self._model = 'faster'  # this is then also valid for YOLO

    def reset(self):
        # self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
        
        
        # self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        # if self._temp_dir != self._working_dir.name:
        #     self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to directory {} ...".format(self._temp_dir)
        )


    def convert_boxes_to_mask(self, array, maskSize):
        arrayInt = array.round().astype(int)
        mask = np.zeros(maskSize, dtype=np.uint8)
        mask[arrayInt[1]:arrayInt[3]+1, arrayInt[0]:arrayInt[2]+1] = 1
        return mask

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label
        from detectron2.utils.visualizer import Visualizer
        
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")
            
            if 'cityscapes' in file_name or 'leftImg' in file_name:
                # v = Visualizer(input['image'][:, :, ::-1], MetadataCatalog.get('cityscapes_fine_instance_seg_train'), scale=1.0)
                tmp = input['image']
                tmp = torch.transpose(tmp, 0, 1)
                tmp = torch.transpose(tmp, 1, 2)
                v = Visualizer(tmp.numpy(), MetadataCatalog.get('cityscapes_fine_instance_seg_train'), scale=1.0)
                v = v.draw_instance_predictions(output["instances"].to("cpu"))
                image = v.get_image()[:, :, ::-1]
                png_filename = os.path.join(self._temp_dir, basename + "_result.png")
                Image.fromarray(image).save(png_filename)
            
            output = output["instances"].to(self._cpu_device)
            num_instances = len(output)
            with open(pred_txt, "w") as fout:
                for i in range(num_instances):
                    pred_class = output.pred_classes[i]
                    classes = self._metadata.thing_classes[pred_class]
                    class_id = name2label[classes].id
                    score = output.scores[i]

                    if not 'pred_masks' in output._fields:      # then it has to be Faster R-CNN; thus, convert to bboxes
                        mask = self.convert_boxes_to_mask(output.pred_boxes.tensor.numpy()[i, :], output.image_size)
                        self._model = 'faster'
                    else:
                        mask = output.pred_masks[i].numpy().astype("uint8")
                        self._model = 'mask'
                        
                    png_filename = os.path.join(
                        self._temp_dir, basename + "_{}_{}.png".format(i, classes)
                    )

                    Image.fromarray(mask * 255).save(png_filename)
                    fout.write("{} {} {}\n".format(os.path.basename(png_filename), class_id, score))

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        os.environ["CITYSCAPES_DATASET"] = os.path.abspath(
            os.path.join(self._metadata.gt_dir, "..", "..")
        )
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = True
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")
        cityscapes_eval.args.exportFile = self._output_json_file
        
        # check whether masks have to be converted to bboxes or not
        if self._model == 'faster':
            cityscapes_eval.args.convertGtMasks = True
            cityscapes_eval.args.convertPredMasks = False
        elif self._model == 'bbox':
            cityscapes_eval.args.convertGtMasks = True
            cityscapes_eval.args.convertPredMasks = True
        elif self._model == 'mask':
            cityscapes_eval.args.convertGtMasks = False
            cityscapes_eval.args.convertPredMasks = False
        else:
            print('Unknown Network. Assuming no conversion to bounding boxes')
            cityscapes_eval.args.convertGtMasks = False
            cityscapes_eval.args.convertPredMasks = False
            
        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        groundTruthImgList = glob.glob(cityscapes_eval.args.groundTruthSearch)
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        groundTruthImgList.sort()
        
        ################################## Modify for reducing number of images ########################################
        # groundTruthImgList = groundTruthImgList[:5]
        # make sure that the function load_cityscapes_instances() from file cityscapes.py (line 67) fits to the settings here
        ################################################################################################################
        
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
        # self._working_dir.cleanup()
        return ret
 
