"""Configuration file for setting default arguments.

This is where default configurations should be set
and updated.
"""

from modules import icon_finder_shape_context

IOU_THRESHOLD = 0.15
TFRECORD_PATH = "benchmark_single_instance.tfrecord"
FIND_ICON_OBJECT = icon_finder_shape_context.IconFinderShapeContext()
OUTPUT_PATH = ""
