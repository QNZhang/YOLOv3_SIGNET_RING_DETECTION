# Installation
1. Follow the installation instructions from [README.md](README.md)

# Generate img_roiX json and xml files for training
1. Place your kfb images and json files in the `input` folder (create the folder if necessary).

2. Run the lines

   `from utils.files import generate_roi_and_bboxes_files`

   `generate_roi_and_bboxes_files()`
