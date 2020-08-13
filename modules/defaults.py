"""Configuration file for setting default arguments.

This is where default configurations should be set
and updated.
"""

from modules import icon_finder_shape_context

IOU_THRESHOLD = 0.15
FIND_ICON_OBJECT = icon_finder_shape_context.IconFinderShapeContext()
TFRECORD_PATH = "datasets/benchmark_single_instance.tfrecord"
FIND_ICON_OPTION = "shape-context"
OUTPUT_PATH = ""
