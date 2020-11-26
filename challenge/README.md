# Signet Ring Cell Detection Challenge task 1

<p align="left"><img src="../output_sample/2018_64982_1-3_2019-02-25%2021_57_36-lv0-36515-58465-2013-2071.jpeg" height="620"\></p>

## Installation
The code to process WSI jpeg images is located at **challenge** folder. Thus,
The procedure to analise WSI jpeg images is the following:

1. Follow the installation instructions from [README.md](../README.md) and [README_SIGNET_RING_DETECTION.md](../README_SIGNET_RING_DETECTION.md).
2. (Optional step if not using the docker container) Make sure to also install
   the requirements of this project (we do recommend using a virtual environment)
   
   `cd ..`
   `pip install -r requirements.txt`
2. Go to **challenge** folder, make a copy of the settings file and rename it as
   `settings.py`
2. Open the `settings.py` file and modify it as necessary.
3. Optionally, open the `config/yolov3_eval_digestpath.cfg` file and modify as
   necessary the confidence threshold (CONFTHRE), non maximum suppression
   threshold (NMSTHRE) and image size (IMGSIZE) from TEST section.
4. Create an folder called **input** at project's root and place there the WSI jpeg
   images to be analysed.
5. Run the [Segmentation.py](../Segmentation.py) file to get the predictions at
   the output folder in the project's directory.
   
   `python Segmentation.py`

## Quick test
Using provided checkpoint and sample images

1. Rename **input_sample** folder.
     
     `cp -r input_sample input`

2. Get sample checkpoint (if not already done). Download it at the same folder were input_sample is located.

     `git clone git@github.com:giussepi/PyTorch_YOLOv3_sample_checkpoint.git`

3. Rename `PyTorch_YOLOv3_sample_checkpoint` folder (if not already done).

    `mv PyTorch_YOLOv3_sample_checkpoint checkpoints`
    
4. Run the `Segmentation.py`.

     `python Segmentation.py`

5. Open and review the **output** folder
