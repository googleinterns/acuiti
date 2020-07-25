"""This module contains a BoundingBox class."""
import dataclasses


@dataclasses.dataclass(eq=True, frozen=True)
class BoundingBox:
  """Class for keeping track of a bounding box's coordinates.

   Assumes each coordinate corresponds to a pixel, and
    therefore is zero-indexed.
  """
  min_x: int
  min_y: int
  max_x: int
  max_y: int

  def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float):
    object.__setattr__(self, "min_x", int(min_x))
    object.__setattr__(self, "min_y", int(min_y))
    object.__setattr__(self, "max_x", int(max_x))
    object.__setattr__(self, "max_y", int(max_y))

  def calculate_area(self) -> float:
    # add one in calculations because pixel numbers are 0-indexed
    return (self.max_x - self.min_x + 1) * (self.max_y - self.min_y + 1)

  def calculate_iou(self, other_box: 'BoundingBox') -> float:
    """Calculate the intersection over union of self and other bounding box.

    The intersection is the overlap of two bounding boxes,
    and the union is the total area of two bounding boxes.

    Arguments:
      other_box: other BoundingBox.

    Returns:
      float -- intersection over union of the two bounding boxes.
    """
    overlap_box = BoundingBox(max(self.min_x, other_box.min_x),
                              max(self.min_y, other_box.min_y),
                              min(self.max_x, other_box.max_x),
                              min(self.max_y, other_box.max_y))

    if overlap_box.max_x < overlap_box.min_x or overlap_box.max_y < overlap_box.min_y:
      return 0.0

    intersection_area = overlap_box.calculate_area()
    self_box_area = self.calculate_area()
    gold_box_area = other_box.calculate_area()
    iou = intersection_area / float(self_box_area + gold_box_area -
                                    intersection_area)
    return iou
