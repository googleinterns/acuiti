"""This module has a BBGenerator class for generating bounding boxes.

 This is where ShapeContextDescriptor algorithm would go after implemented.
"""
import random
from typing import List

from modules.bounding_box import BoundingBox


class BBGenerator:
  """This class has different methods to generate bounding boxes."""

  def __init__(self, image_list: List[BoundingBox],
               icon_list: List[BoundingBox]):
    self.image_dataset = image_list
    self.icon_dataset = icon_list

  def generate_random(self) -> List[BoundingBox]:
    """Generates a list of random bounding boxes corresponding to the images in the dataset."""
    bb_list = []
    for image_bgr in self.image_dataset:
      height = image_bgr.shape[0]
      width = image_bgr.shape[1]
      # using 0-indexed pixel numbers
      min_y = random.randint(0, height - 1)
      min_x = random.randint(0, width - 1)
      max_x = random.randint(min_x, width - 1)
      max_y = random.randint(min_y, height - 1)
      bb = BoundingBox(min_x, min_y, max_x, max_y)
      bb_list.append(bb)
    return bb_list
