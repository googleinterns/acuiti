"""This module contains a BoundingBox class."""
from dataclasses import dataclass


@dataclass
class BoundingBox:
  """Class for keeping track of a bounding box's coordinates.

   Assumes each coordinate corresponds to a pixel, and
    therefore is zero-indexed.
  """
  min_x: float
  min_y: float
  max_x: float
  max_y: float

  def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float):
    self.min_x = min_x
    self.min_y = min_y
    self.max_x = max_x
    self.max_y = max_y

  def calculate_area(self) -> float:
    # add one in calculations because pixel numbers are 0-indexed
    return (self.max_x - self.min_x + 1) * (self.max_y - self.min_y + 1)
