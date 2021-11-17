#!/usr/bin/env bash

#sleep $[ ( $RANDOM % 10 ) ].$[ ( $RANDOM % 10 ) + 1 ]s

mkdir -p ./cluster_done

echo $JOB_ID
echo $HOSTNAME

nvidia-smi

echo /home/fischer/anaconda3/envs/detectron2/bin/python3.8 scripts/keypoints/run_evaluation_keypoints.py ${params}
/home/fischer/anaconda3/envs/detectron2/bin/python3.8 scripts/keypoints/run_evaluation_keypoints.py ${params}


find .  -maxdepth 1 -type f -name "*$JOB_ID" -exec mv -t ./cluster_done/ {} +
exit