 # FP Liu

Repository for research project Sicheng Liu "Labelling and Evaluating DIV2K Dataset for Human Keypoint Detection on Compressed Images"

This repository should be used to save and share current coding progress.

### Links and Exemplary Code

* PyTorch framework for object detection https://github.com/facebookresearch/detectron2 (can also be used for Keypoint detection)
    * Models for keypoint detection can be downloaded here: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md

* *Additional_Code_Detectron2* is an example on how to run an R-CNN with Cityscapes data; call_qsubs_detectron2.sh has to be pushed to back node at lms37-27 GPU-Cluster in order to run the run_evaluation.py script

* *DIV2K* Dataset can be found under ~/SHARED_FILES/CLUSTER_SEQUENCES/HD/DIV2K/

* Examples for labeling COCO dataset with keypoints is provided under https://cocodataset.org/#keypoints-2020

* Annotation tool can be found here: https://github.com/jsbroks/coco-annotator

### Working on Clutser

* Cluster Wiki see *README_cluster.md*

* CPU-Cluster

        ssh lntcluster

* GPU-Cluster
    
        ssh lms37-27
        
    - qsub:   folgende Variablen können bzw. müssen beim Aufruf von qsub angegeben werden.
    
            gpu=X ( zwischen 0 und 3) – Variable muss angegeben werden
    
            gpumem=X ( max. 11000) entspricht der größten Karte mit 11GByte Memory  - kann mit angegeben werden
    
     - Examples:
     
            qsub -cwd -l gpu=1 -l gpumem= 6000   jobscript ( job geht an eine Maschine mit mindestens 1 GPU und 6GByte Memory)
    
            qsub  -l gpu=2  jobscript ( job geht an Maschine mit mindestens 2GPUs  )
    
            qsub -l gpu=0 ( job geht auf Maschine mit freien CPU-Slot- dadurch können freie CPU-Resourcen genutzt werden)
            
     - Additional information:

			Hallo GPU-User,

			ihr könnt nun ab sofort den GPU-Cluster testen.

			Hostname: lntgpucluster.e-technik.uni-erlangen. de

			User quota: 1000GByte

			Nodes: NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1 / kein MATLAB

			FrontNode: No Nvidia / kein MATLAB

			OS: Alle Centos 7.6

			

			Es gibt im Moment 5 Nodes:

			compute-0-2: 3 GPUs mit jeweils 11GByte; 4 CPU-Cores/ 4 Slots

			compute-0-3: 2 GPU mit 12 GByte Memory; 20 CPU-Cores/ 10Slots

			compute-0-4: 4 GPU mit 11GByte Memory;  24 CPU-Cores/ 10 Slots

			compute-0-5: 2 GPU mit 11GByte Memory;  24 CPU-Cores/ 10 Slots

			compute-0-7: 2 GPU mit 8 GByte Memory;  20 CPU-Cores/ 10 Slots

			

			Cuda Compute Capability reicht von 61 bis 75 ( entspricht 6.1 bis 7.5)

			

			qsub:   folgende Variablen können bzw. müssen beim Aufruf von qsub angegeben werden.

							gpu=X ( zwischen 0 und 4) – Variable muss angegeben werden

							gpumem=X ( max. 12000) entspricht der größten Karte mit 12GByte Memory - kann mit angegeben werden

							gpucc=XX ( z.B. 61) entspricht der Cuda Compute Capability ohne Dezimal Punkt – kann mit angegeben werden

			z.B          qsub -cwd -l gpu=1 -l gpumem=6000   jobscript ( job geht an eine Maschine mit mindestens 1 GPUund 6GByte Memory)

							qsub  -l gpu=2  jobscript ( job geht an Maschine mit mindestens 2GPUs  )

							qub -l gpu=0 ( job geht auf Maschine mit freien CPU-Slot - dadurch können freie CPU-Resourcen genutzt werden)

			

			Achtung: wird „-l gpu=0“ angegeben und im Code doch eine GPU angesprochen, stürzt wahrscheinlich entweder das eigene Programm ab ( wenn keine freie GPU verfügbar ist ) oder der nachfolgend gestartete Job, welcher eine frei GPU erwartet und keine mehr bekommt ( nur ein Thread pro GPU). Werden mehr als 4 GPUs angefordert startet der Job gar nicht, da keine Maschine mehr als 4 GPUs hat.

			Wichtig: Job Umgebungsvariable „CUDA_VISIBLE_DEVICES“

			Die Variable CUDA_VISIBLE_DEVICES wird vom System gesetzt und gibt die zu verwendenden Kartenummern an.( Siehe nvidia-smi)

			Wird zum Beispiel ein Job mit einer GPU angefordert:

			qsub -l gpu=1 ../job_was_weiss_ich

			und wird auf einem System mit 3 GPUs ( Kartennummer 0,1,2 ) gestartet,

			so kann die Variable den Wert 0 oder 1 oder 2 annehmen. Je nachdem ob schon weitere Jobs laufen.

			Das User-Programm muss dann CUDA_VISIBLE_DEVICES auslesen und darf dann nur diese Karte mit der Nummer verwenden.

			Werden zwei GPUs angefordert, so kann die Variable z.B. den Wert "1,2" annehmen.

 
    
        
