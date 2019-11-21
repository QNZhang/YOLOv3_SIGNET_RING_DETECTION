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

### Create JPEG of minipatches along with their bounding boxes (optional)
1. Review the function definition of `utils.plot_tools.create_X_cervical_images_plus_bounding_boxes`

2. Then run the lines (modify them as necessary)

   `from utils.plot_tools import create_X_cervical_images_plus_bounding_boxes`

   `create_X_cervical_images_plus_bounding_boxes((0, 100))`

### Plot image along with its bouding boxes
1. Review `utils.plot_tools.plot_img_plus_bounding_boxes`

2. Run the following lines (modify them as necessary)

	`from utils.plot_tools import plot_img_plus_bounding_boxes`

	`plot_img_plus_bounding_boxes('T2019_999-roi2_33232_15350.jpeg', save_to_disk=True)`


### Testing

1. Set the model checkpoint, yolov3 config file, test input folder (containing kfb ), test output folder and test tmp data properly in your settings file, at Tianchi Testing section.


2. Create test jpeg minipatches (it'll take a while)

	`from utils.classes.cutpatch import TestMiniPatch`

	`TestMiniPatch()()`

3. Analyse and create predictions

	`from tianchi_challenge.classes import MyModel`

    `from tianchi_challenge.utils import initial_validation_cleaning, process_input_files`

    `initial_validation_cleaning()`

    `model = MyModel()`

    `process_input_files(model)`
