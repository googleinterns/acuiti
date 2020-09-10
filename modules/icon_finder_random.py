"""This module has an IconFinderRandom class for randomly finding bounding boxes.
"""
import random
from typing import List, Optional, Tuple

import memory_profiler

from modules.bounding_box import BoundingBox
import modules.icon_finder
import numpy as np


class IconFinderRandom(modules.icon_finder.IconFinder):  # pytype: disable=module-attr
  """This class generates bounding boxes randomly."""
  @memory_profiler.profile

  def find_icons(
      self, image: np.ndarray, icon: np.ndarray
  ) -> Tuple[List[BoundingBox], Optional[List[np.ndarray]],
             Optional[List[np.ndarray]]]:
    """Find instances of icon in a given image randomly.

    Arguments:
        image: Numpy array representing image
        icon: Numpy array representing icon

    Returns:
        Tuple[Bounding Box for each instance of icon in image,
        empty list,
        empty list]. The empty lists are just to satisfy the overall
        IconFinder API but would normally contain
    """
    height = image.shape[0]
    width = image.shape[1]
    # Using 0-indexed pixel numbers
    min_y = random.randint(0, height - 1)
    min_x = random.randint(0, width - 1)
    max_x = random.randint(min_x, width - 1)
    max_y = random.randint(min_y, height - 1)
    return [BoundingBox(min_x, min_y, max_x, max_y)], None, None
