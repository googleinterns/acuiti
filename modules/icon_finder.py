"""This module contains the IconFinder base class.
"""
import abc
from typing import List

from modules.bounding_box import BoundingBox
import numpy as np


class IconFinder(abc.ABC):
  """IconFinder is the base class for all IconFinder classes.
  """

  @abc.abstractmethod
  def find_icons(self, image: np.ndarray,
                 icon: np.ndarray) -> List[BoundingBox]:
    """Find instances of icon in image.

    Arguments:
        image: Numpy array representing image
        icon: Numpy array representing icon

    Returns:
        List[BoundingBox] -- Bounding Box for each instance of icon in image.
    """
    pass
