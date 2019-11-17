# Installation
1. Follow the installation instructions from [README.md](README.md)

# Training
## Generate img_roiX json and xml files for training
1. Place your kfb images and json files in the `input` folder (create the folder if necessary).

2. Run the lines

   `from utils.files import generate_roi_and_bboxes_files`

   `generate_roi_and_bboxes_files()`


## Generate minipatches
1. Rename `input` and `output` folder from previous step

   `mv input input_a`

   `mv output output_a`

2. Create a new `input` folder and move the content from `output_a` to this new folder (or just rename `output_a` to `input` ).

3. Copy (or move) the kfb images from `input_a` to `input` folder.

   `cd input_a`

   `rsync -zarv --include "*/" --include="*.kfb" --exclude="*" . ../input`

   OR

   `mv *.kfb ../input`

4. Run the lines

   `from utils.classes.cutpatch import MiniPatch`

   `MiniPatch()()`

### Create JPEG of minipatches along with their bounding boxes
1. Review the function definition of `utils.plot_tools.create_X_cervical_images_plus_bounding_boxes`

2. Then run the lines (modify them as necessary)

   `from utils.plot_tools import create_X_cervical_images_plus_bounding_boxes`

   `create_X_cervical_images_plus_bounding_boxes((0, 100))`
