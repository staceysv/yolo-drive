#! /usr/bin/env python
import argparse
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from numpy import expand_dims
import numpy as np
import os
import PIL

import wandb
import time

from bb_utils import decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes
from yolo import make_yolov3_model, WeightReader


# Parameters used in the Dataset, on which YOLOv3 was pretrained
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# define the expected input shape for the model
WIDTH, HEIGHT = 416, 416

# define the expected output shape for logging
LOG_WIDTH, LOG_HEIGHT = 1200, 800

# define the probability threshold for detected objects
class_threshold = 0.9

# load and prepare an image
def load_image_pixels(filename, shape=(WIDTH, HEIGHT)):
  '''
  Function preprocess the images to 416x416, which is the standard input shape for YOLOv3, 
  and also keeps track of the original image shape, which is later used to draw the boxes.
  parameters:
    filename {String}: path to the image
    shape {tuple}: shape of the input dimensions of the network
    
  returns:
    image {PIL}: image of shape 'shape'
    width {int}: original width of the picture
    height {int}: original height of the picture
  '''
  # load the image to get its original shape
  image = load_img(filename)
  width, height = image.size
    
  # load the image with the required size
  image = load_img(filename, target_size=(HEIGHT, WIDTH))
    
  # convert to numpy array
  image = img_to_array(image)
    
  # scale pixel values to [0, 1]
  image = image.astype('float32')
  image /= 255.0
    
  # add a dimension so that we have one sample
  image = expand_dims(image, 0)
  return image, width, height

def run_experiment(cfg):
  default_config = {
    "pretrained" : cfg.pretrain,
    "class_threshold" : class_threshold
  }

  wandb.init(project="yolo-drive", entity="stacey", name=cfg.model_name, config=default_config)
  
  # load pretrained model for now
  if cfg.pretrain:
    model = load_model("model.h5", compile=False)
  else:
    model = make_yolov3_model(cfg)

  # load images
  sample_images = []
  image_path = "../sample_data/"
  for i in os.listdir(image_path):
    filename = image_path + i
    sample_images.append(filename)
   
  all_boxes = []
  all_pyplot = []
  for img_path in sample_images:
    print("loading image: ", img_path) 
    
    # load picture with old dimensions
    image, image_w, image_h = load_image_pixels(img_path)
    
    # Predict image
    yhat = model.predict(image)
    
    # Create boxes
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, HEIGHT, WIDTH)
    
    # the input dataset is of highly variable size
    # the model trains and preicts with image size 416 x 416
    # the initial boxes are based on this original image size
    # we log the examples in a standard size: 1200 x 800
    # correct the sizes of the bounding boxes for the resulting shape of the image
    correct_yolo_boxes(boxes, LOG_HEIGHT, LOG_WIDTH, HEIGHT, WIDTH)

    # suppress non-maximal boxes
    do_nms(boxes, 0.5)

    
    # define the labels (subset of YOLOv3 labels relevant to this task)
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", \
              "truck", "boat", "traffic light", "fire hydrant", "stop sign", \
               "parking meter", "bench"]

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    # summarize the detections
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    # draw what we found
    wb_image, pyplot_fig = draw_boxes(img_path, v_boxes, v_labels, v_scores, LOG_WIDTH, LOG_HEIGHT)
    all_boxes.append(wb_image)
    all_pyplot.append(pyplot_fig)
  
  # buffer time to log full images
  time.sleep(5)
  wandb.log({"2D bounding boxes" : all_boxes, "pyplot figures" : all_pyplot})
    
# for eventual re-training
  
# define the model
#model = make_yolov3_model()

# load the model weights
# I have loaded the pretrained weights in a separate dataset
#weight_reader = WeightReader('../input/lyft-3d-recognition/yolov3.weights')

# set the model weights into the model
#weight_reader.load_weights(model)

# save the model to file
#model.save('model.h5')
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default="",
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  parser.add_argument(
    "-p",
    "--pretrain",
    action="store_true",
    help="Use saved model, do not retrain")

  args = parser.parse_args()
  run_experiment(args)
