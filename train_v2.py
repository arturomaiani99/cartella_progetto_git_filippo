"""
This scripts loads a trained backbone of your choice and start training these weights/last training session.
"""
print("Importing libraries...")
import os
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
import os
import json
import matplotlib.pyplot as plt

from detectron2.utils.logger import setup_logger

# import some common detectron2 utilities
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import hooks
from detectron2.engine import DefaultTrainer

import sys
import numpy
register_once = 1
DEVICE = "0"
GPU_ORDER = "PCI_BUS_ID"

epochs = 1000
num_images = 1389
experiment_version = '3'

model_name = 'mask_rcnn_R_50_FPN_3x'

experiment_dir  = "/home/insoore/insoore_team/dev/damageDetection/training/modelChechpointsAndLogs"

output_dir = os.path.join(experiment_dir, f'experimentV{experiment_version}_e{epochs}_n{num_images}_{model_name}')

#MODEL SETUP
def setup():
    """
    Create configs and perform basic setups.
    """ 
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{model_name}.yaml"))

    cfg.MODEL.WEIGHTS = '/home/insoore/backup230117/body_parts_training/detectron2/damages_v6/model_final.pth'
    
    if DEVICE == "0":
        cfg.MODEL.DEVICE = "cuda"
        print("Device: GPU")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("Device: CPU")
    
    cfg.DATASETS.TRAIN = ("pandas2kTrain",)
    cfg.DATASETS.TEST = ("pandas2kVal",)
    
    # cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    cfg.DATALOADER.NUM_WORKERS = 6
    
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.STEPS = [5000, 35000, 100000, 135000, 150000]
                        # 0.001   0.0001  0.00001 1e-5
    
    cfg.SOLVER.MAX_ITER = epochs * int(np.ceil(num_images / cfg.SOLVER.IMS_PER_BATCH))
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    
    cfg.OUTPUT_DIR = output_dir

    return cfg
#############################################################
setup_logger(output=output_dir)


os.environ["CUDA_DEVICE_ORDER"] = GPU_ORDER
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

cfg = setup()


partial_path = os.getcwd().rsplit("/", 1)[0]
sys.path.insert(0, f"{partial_path}/src/")

if register_once == 1:
    register_coco_instances("pandas2kTrain", {}, 
                            "/home/insoore/insoore_team/datasets/pandas2k/allClasses/trainAnnotations.json", 
                            "/home/insoore/insoore_team/datasets/pandas2k/pandas2kAnnoatatedFinalExifCleaned")
    register_coco_instances("pandas2kVal", {}, 
                            "/home/insoore/insoore_team/datasets/pandas2k/allClasses/valAnnotations.json", 
                            "/home/insoore/insoore_team/datasets/pandas2k/pandas2kAnnoatatedFinalExifCleaned")
    register_once = 0

# my_dataset_train_metadata = MetadataCatalog.get("pandas2kTrain")


print('Training is initiated on:', cfg.MODEL.DEVICE)
print('Number of epochs:', epochs)
print('Number of train images:', num_images)
print('Number of max iterations:', cfg.SOLVER.MAX_ITER)
print(f'A checkpoint of the model is being saved each {cfg.SOLVER.CHECKPOINT_PERIOD} iterations.')
print('Watch the live training of the model on tensorboard...')

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 

# trainer.test(cfg, DefaultPredictor(cfg).model, COCOEvaluator("pandas2kVal", output_dir="./output"))

trainer.resume_or_load(resume=True)

trainer.train()

print(f"Training completed! Best model saved at {cfg.OUTPUT_DIR}")