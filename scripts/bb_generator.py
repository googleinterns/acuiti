"""This module has a BBGenerator class."""
import random
from bb import BoundingBox
import cv2
import numpy as np


class BBGenerator:
  """This class has different methods to generate bounding boxes."""

  def __init__(self, parsed_image_dataset):
    self.image_dataset = parsed_image_dataset

  def generate_random(self):
    """Generates a list of random bounding boxes corresponding to the images in the dataset."""
    bb_list = []
    for image_features in self.image_dataset:
      bb = {}
      image_raw = image_features["encoded_image_png"].numpy()
      image_bgr = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), -1)
      height = image_bgr.shape[0]
      width = image_bgr.shape[1]
      min_y = random.randint(0, height - 1)
      min_x = random.randint(0, width - 1)
      max_x = random.randint(min_x, width - 1)
      max_y = random.randint(min_y, height - 1)
      bb = BoundingBox(min_x, min_y, max_x, max_y)
      bb_list.append(bb)
    return bb_list
