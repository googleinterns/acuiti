"""Configuration file for setting default arguments.

This is where default configurations should be set
and updated.
"""

from modules import icon_finder_shape_context


FIND_ICON_OBJECT = icon_finder_shape_context.IconFinderShapeContext()
IOU_THRESHOLD = 0.6
TFRECORD_PATH = "datasets/small_single_instance_v2.tfrecord"
OUTPUT_PATH = ""
