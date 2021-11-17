import os
from scripts.keypoints.run_evaluation_keypoints import run_evaluation
from types import SimpleNamespace
import socket

QP = [0, 12, 17, 22, 27, 32, 37]
# QP = [0]

CONFIDENCE_THRESHOLD = 0.3

hostname = socket.gethostname()

CODECS = ['VTM-10.0', 'HM-16.18']
# CODECS = ['VTM-10.0']


# DEFAULT_PATH = 'datasets/cityscapes/'
DEFAULT_PATH = 'datasets/DIV2K_keypoints/'

## Keypoint R-CNN
CONFIG_FILE = 'configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
MODEL_PATH = 'models/COCO/keypoint_rcnn_R_50_FPN_x3/model_final_a6e10b.pkl'

# ## Faster R-CNN COCO
# CONFIG_FILE = 'configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
# MODEL_PATH = 'models/COCO/faster_rcnn_R_50_FPN_x3/model_final_280758.pkl'

## Faster R-CNN Cityscapes
# CONFIG_FILE = 'configs/Cityscapes/faster_rcnn_R_50_FPN_3x.yaml'
# MODEL_PATH = 'models/Cityscapes/faster_rcnn_R_50_FPN/model_final_self_trained_correct.pth'  # self trained wAP uncompressed of ~0.48 on Cityscapes validation


# INPUT_BASE_PATH = '/CLUSTERHOMES/fischer/cityscapes_vtm_frdo/leftImg8bitCompressedFRDO/' if hostname == 'lms40-127' else '/home/fischer/test_images_vtm-10.0/'
INPUT_BASE_PATH = os.path.join('/home/fischer/ImageDatasets/DIV2K_compressed') if hostname == 'lms40-127' else '/home/fischer/DIV2K_compressed/'
# ORIG_PATH = '/CLUSTERHOMES/fischer/cityscapes_vtm/leftImg8bit/' if hostname == 'lms40-127' else '/home/fischer/cityscapes_data/leftImg8bit'
ORIG_PATH = os.path.join(DEFAULT_PATH, 'DIV2K_images') if hostname == 'lms40-127' else '/home/fischer/sequences/HD/DIV2K/DIV2K_train_HR/'
RESULT_FILE_BASE = 'resultKeypointDetection'
GT_INSTANCE_PATH = os.path.join(DEFAULT_PATH, 'labels', 'test_keypoint.json')
GT_INSTANCE_PATH = os.path.join(DEFAULT_PATH, 'labels', 'test_keypoint_with_crowd_converted.json')


def main():

    optList = ['MODEL.WEIGHTS']
    optList.append(MODEL_PATH)
    
    modelName = MODEL_PATH.split('/')[-2] + '/'

    counter = 0
    for cI, codec in enumerate(CODECS):
        for qp in QP:
            if qp == 0:
                if not cI == 0:     # you only have to run uncompressed case once
                    continue
                suffix = '_uncompressed'
                qpPath = 'uncompressed'
                # codec = 'uncompressed'

                inputPath = ORIG_PATH
                outputPathJson = os.path.join(DEFAULT_PATH, 'evaluationResults', 'uncompressed', modelName)
                outputPathResults = os.path.join(DEFAULT_PATH, 'results', 'uncompressed', modelName)
                # outputPathResults = ''
            else:
                suffix = '_qp_' + str(qp)
                qpPath = 'qp_' + str(qp)

                inputPath = os.path.join(INPUT_BASE_PATH, codec, qpPath)
                outputPathJson = os.path.join(DEFAULT_PATH, 'evaluationResults', codec, modelName)
                outputPathResults = os.path.join(DEFAULT_PATH, 'results', codec, modelName, qpPath)

            resultJsonFile = os.path.join(outputPathJson, RESULT_FILE_BASE + suffix + '.txt')
            if os.path.exists(resultJsonFile):
                if os.path.getsize(resultJsonFile) > 0:
                    print('Output file %s exists already' % resultJsonFile)
                    continue

            if hostname == 'lms40-127':
                args = SimpleNamespace(configFile=CONFIG_FILE,
                                       outputPathJson=outputPathJson,
                                       opts=optList,
                                       outputPathClassifiedImages=outputPathResults,
                                       inputPath=inputPath,
                                       outputSuffix=suffix,
                                       gtInstancesPath=GT_INSTANCE_PATH,
                                       confidenceThreshold=CONFIDENCE_THRESHOLD)
                run_evaluation(args)
            else:
                paramString = '"--configFile ' + CONFIG_FILE + \
                        ' --outputPathJson ' + outputPathJson + \
                        ' --outputPathClassifiedImages ' + outputPathResults + \
                        ' --inputPath ' + inputPath + \
                        ' --outputSuffix ' + suffix + \
                        ' --gtInstancesPath ' + GT_INSTANCE_PATH + \
                        ' --confidenceThreshold ' + str(CONFIDENCE_THRESHOLD) + \
                        ' --opts ' + optList[0] + ' ' + optList[1] + '"'
                exeString = 'qsub -cwd -l gpu=1 -q all.q@compute-0-3.local,all.q@compute-0-4.local,all.q@compute-0-5.local,all.q@compute-0-7.local -v params=' + \
                            paramString + ' scripts/keypoints/call_qsubs_detectron2_keypoints.sh'
                # exeString = 'qsub -cwd -l gpu=1 -l gpumem=8500 -v params=' + paramString + ' scripts/call_qsubs_detectron2.sh'
                print(exeString)
                os.system(exeString)
                counter += 1
                # time.sleep(60)   # wait 1 second so that cluster can assign the available GPUs properly
    print('Pushed %i jobs to cluster' % counter)


if __name__ == "__main__":
    main()

