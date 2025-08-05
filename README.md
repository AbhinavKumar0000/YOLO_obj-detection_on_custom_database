# YOLO Object Detection with Custom Dataset Training
Description
This project implements real-time object detection using the Ultralytics YOLO model. It includes a script for performing detection on live camera feeds or video files, saving the annotated output, and a comprehensive workflow for training a custom YOLO model on a user-provided dataset. The training process involves data preparation, configuration file generation, and leverages NVIDIA GPUs for accelerated training.

Features
Real-time Object Detection: Detect objects from webcam or video files using a pre-trained or custom YOLO model.

Custom Model Training: Train a YOLOv11s model on your own dataset, with automated data splitting and configuration.

Annotated Video Output: Save detection results to an output video file with bounding boxes and labels.

Configurable Input/Output: Easily specify input source (webcam, USB camera, video file path) and output file path.

GPU Acceleration: Utilizes NVIDIA GPUs for efficient model training and inference.

Class Filtering: The detection script is configured to skip specific object classes (e.g., 'Watch') from detection and annotation.

Installation
To get a local copy up and running, follow these simple steps.

Prerequisites
Python 3.x

pip (Python package installer)

NVIDIA GPU with CUDA support (highly recommended for training and faster inference)

unzip utility (for extracting dataset)

Steps
Clone the repository:

git clone https://github.com/your-username/your-project.git
cd your-project

Prepare your custom dataset:

Ensure your dataset is in YOLO format (images and corresponding .txt label files).

Create a data.zip file containing your images and labels.

Unzip your data to a custom data folder (e.g., /content/custom_data if using Colab):

!unzip -q /content/data.zip -d /content/custom_data

Split data into training and validation sets:

Download the train_val_split.py script:

!wget -O /content/train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py

Run the split script, adjusting --datapath and --train_pct as needed:

!python train_val_split.py --datapath="/content/custom_data" --train_pct=0.9

Install Ultralytics:

!pip install ultralytics

Create YOLO configuration file (data.yaml):

This Python function reads your classes.txt (expected in /content/custom_data/classes.txt) and generates the necessary data.yaml configuration.

import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
  # Read class.txt to get class names
  if not os.path.exists(path_to_classes_txt):
    print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
    return
  with open(path_to_classes_txt, 'r') as f:
    classes = []
    for line in f.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())
  number_of_classes = len(classes)

  # Create data dictionary
  data = {
      'path': '/content/data', # Base path where train/val folders are located
      'train': 'train/images', # Relative path to training images
      'val': 'validation/images', # Relative path to validation images
      'nc': number_of_classes, # Number of classes
      'names': classes # List of class names
  }

  # Write data to YAML file
  with open(path_to_data_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
  print(f'Created config file at {path_to_data_yaml}')

# Define path to classes.txt and run function
path_to_classes_txt = '/content/custom_data/classes.txt' # Path to your classes.txt file
path_to_data_yaml = '/content/data.yaml' # Output path for data.yaml
create_data_yaml(path_to_classes_txt, path_to_data_yaml)

Usage
Training a Custom Model
To train your YOLO model, use the following command. Adjust model, epochs, and imgsz as needed based on your dataset size and desired performance.

!yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=60 imgsz=640

data: Path to your data.yaml configuration file (e.g., /content/data.yaml).

model: Base YOLO model to start training from (e.g., yolo11s.pt, yolov8n.pt).

epochs: Number of training epochs.

imgsz: Input image size for training (e.g., 640 for 640x640).

Running Object Detection
After training (or with a pre-trained model), you can run the detection script:

python yolo_detect.py --model path/to/your/model.pt --source 0 --output output_video.mp4

--model: Path to your trained YOLO model file (e.g., runs/detect/train/weights/best.pt after training).

--source: Input source for detection. Use 0 for the default webcam, usb0 for a USB camera, or a path to a video file (e.g., input_video.mp4).

--output: Path to save the output video with detected objects and annotations.

Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

License
Distributed under the MIT License. See LICENSE for more information.

Contact
Your Name - abhinavkumarsaksena@gmail.com

