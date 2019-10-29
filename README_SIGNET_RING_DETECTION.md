<p align="left"><img src="output_sample/2018_64982_1-3_2019-02-25 21_57_36-lv0-33516-59515-2003-2010.jpeg" height="620"\></p>

# Installation

1. Follow the installation instructions from [README.md](README.md)

2. (Optional step if not using the docker container) Make sure to also install
   the requirements of this project (we do recommend using a virtual environment)
   
   `pip install -r requirements.txt`
   
3. Place your signet ring dataset on a suitable folder which can be accessed from this application.

4. Make a copy of the settings template.

   `cp settings_template.py settings.py`
   
5. Open your settings.py and modify the paths for your database properly and train/test pickle files properly (the latter will be created in the next step).

6. Create split your database and create your pickle files

   `from utils.data import get_or_create_train_test_files`
   
   `get_or_create_train_test_files(test_size=0.8, random_state=42, shuffle=True, force_create=True)`
   
7. The configuration files for training and testing are located at:

    `config/yolov3_default_digestpath.cfg`
    
    `config/yolov3_eval_digestpath.cfg`
    
8. If you want to re-calculate the anchor boxes for your dataset you can do it by running the function recalculate_anchor_boxes_kmeans_iou, then you just need to update the ANCHORS on both configuration files.

    `from constants import Dataset`
    
    `from utils.utils import recalculate_anchor_boxes_kmeans_iou`
    
    `new_anchors = recalculate_anchor_boxes_kmeans_iou(Dataset.SIGNET_RING, print_results=True, num_centroids=9)`
    
    `print(new_anchors)`
9. Based on the size of your images and hardware especifications you should update the following variables from the configuration files: `MAXITER, BATCHSIZE, SUBDIVISION, IMGSIZE`.
10. Other customizations could be done on `CONFTHRE and NMSTHRE`.


# Training

`import os`

`os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard True --checkpoint_interval=50 --eval_interval=50')`


# Evaluation

Quick example using our provided checkpoint
 
1. Get sample checkpoint. Download it at the same folder were input_sample is located.

     `git clone git@github.com:giussepi/PyTorch_YOLOv3_sample_checkpoint.git`

2. Rename `PyTorch_YOLOv3_sample_checkpoint` folder.

    `mv PyTorch_YOLOv3_sample_checkpoint checkpoints`
  
3. Run the evaluation command.

    `import os`
    
    `os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --checkpoint "checkpoints/confthre_0_dot_8/snapshot17350.ckpt"')`
