"""This module contains the IconFinder base class.
"""
import abc
from typing import List

from modules.bounding_box import BoundingBox


class IconFinder(abc.ABC):
  """IconFinder is the base class for all IconFinder classes.
  """

  def __init__(self, image_list: List[BoundingBox],
               icon_list: List[BoundingBox]):
    self.image_dataset = image_list
    self.icon_dataset = icon_list

  @abc.abstractmethod
  def find_icons(self) -> List[BoundingBox]:
    pass
