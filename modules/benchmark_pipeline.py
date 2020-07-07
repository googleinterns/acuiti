"""BenchmarkPipeline class and tfRecord utility functions."""

import argparse
from typing import Any, List, Tuple

import cv2
import matplotlib
from modules.bounding_box import BoundingBox
import modules.defaults as defaults
from modules.icon_finder_random import IconFinderRandom
import modules.util
import numpy as np
import tensorflow as tf

_IMAGE_FEATURE_DESCRIPTION = {
    "encoded_image_png": tf.io.FixedLenFeature([], tf.string),
    "encoded_icon_png": tf.io.FixedLenFeature([], tf.string),
    "box_ymin": tf.io.FixedLenFeature([], tf.float32),
    "box_xmin": tf.io.FixedLenFeature([], tf.float32),
    "box_ymax": tf.io.FixedLenFeature([], tf.float32),
    "box_xmax": tf.io.FixedLenFeature([], tf.float32),
}


def _parse_image_function(
    example_proto: tf.python.framework.ops.Tensor) -> Any:
  return tf.io.parse_single_example(example_proto, _IMAGE_FEATURE_DESCRIPTION)


def _parse_image_dataset(
    path: str) -> tf.python.data.ops.dataset_ops.MapDataset:
  raw_dataset = tf.data.TFRecordDataset(path)
  parsed_image_dataset = raw_dataset.map(_parse_image_function)
  return parsed_image_dataset


def _parse_gold_boxes(
    parsed_image_dataset: tf.python.data.ops.dataset_ops.MapDataset
) -> List[BoundingBox]:
  """Retrieve a list of bounding boxes from dataset.

  Arguments:
      parsed_image_dataset: Original dataset read from TFRecord

  Returns:
      List of ground truth BoundingBoxes.
  """
  gold_boxes = []
  for image_features in parsed_image_dataset:
    gold_box = BoundingBox(image_features["box_xmin"],
                           image_features["box_ymin"],
                           image_features["box_xmax"],
                           image_features["box_ymax"])
    gold_boxes.append(gold_box)
  return gold_boxes


def _parse_images_and_icons(
    parsed_image_dataset: tf.python.data.ops.dataset_ops.MapDataset
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  """Private function for parsing images and icons.

  Arguments:
      parsed_image_dataset: image dataset read from TFRecord

  Returns:
      (List of images, List of icons)
  """
  image_list = []
  icon_list = []
  for image_features in parsed_image_dataset:
    image_raw = image_features["encoded_image_png"].numpy()
    image_bgr = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), -1)
    image_list.append(image_bgr)
    icon_raw = image_features["encoded_icon_png"].numpy()
    icon_bgr = cv2.imdecode(np.frombuffer(icon_raw, dtype=np.uint8), -1)
    icon_list.append(icon_bgr)
  return image_list, icon_list


class BenchmarkPipeline:
  """Represents a pipeline to test generated Bounding Boxes.

  Usage example:
    benchmark = BenchmarkPipeline("benchmark.tfrecord")
    benchmark.find_icons()
    benchmark.evaluate()
  """

  def __init__(self, tfrecord_path: str = defaults.TFRECORD_PATH):
    parsed_image_dataset = _parse_image_dataset(tfrecord_path)
    self.gold_boxes = _parse_gold_boxes(parsed_image_dataset)
    images, icons = _parse_images_and_icons(parsed_image_dataset)
    self.image_list = images
    self.icon_list = icons
    self.proposed_boxes = []

  def visualize_bounding_boxes(self, output_name: str,
                               boxes: List[BoundingBox]):
    """Visualizes bounding box of icon in its source image.

    Arguments:
        output_name: prefix of filename images should be saved as
        boxes: list of BoundingBoxes
    """
    for i, image_bgr in enumerate(self.image_list):
      box = boxes[i]
      # top left and bottom right corner of rectangle
      cv2.rectangle(image_bgr, (box.min_x, box.min_y), (box.max_x, box.max_y),
                    (0, 255, 0), 3)
      image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

      if image_rgb is None:
        print("Could not read the image.")
      matplotlib.pyplot.imshow(image_rgb)
      matplotlib.pyplot.imsave(output_name + str(i) + ".jpg", image_rgb)

  @staticmethod
  def calculate_iou(proposed_box: BoundingBox, gold_box: BoundingBox) -> float:
    """Calculate the intersection over union of two bounding boxes.

    The intersection is the overlap of two bounding boxes,
    and the union is the total area of two bounding boxes.

    Arguments:
      proposed_box: calculated BoundingBox.
      gold_box: ground truth BoundingBox.

    Returns:
      float -- intersection over union of the two bounding boxes.
    """
    overlap_box = BoundingBox(max(proposed_box.min_x, gold_box.min_x),
                              max(proposed_box.min_y, gold_box.min_y),
                              min(proposed_box.max_x, gold_box.max_x),
                              min(proposed_box.max_y, gold_box.max_y))

    if overlap_box.max_x < overlap_box.min_x or overlap_box.max_y < overlap_box.min_y:
      return 0.0

    intersection_area = overlap_box.calculate_area()
    proposed_box_area = proposed_box.calculate_area()
    gold_box_area = gold_box.calculate_area()
    iou = intersection_area / float(proposed_box_area + gold_box_area -
                                    intersection_area)
    return iou

  def find_icons(
      self,
      find_icon_option: str = defaults.FIND_ICON_OPTION,
      output_path: str = defaults.OUTPUT_PATH) -> Tuple[float, float]:
    """Runs an icon-finding algorithm under timed and memory-tracking conditions.

    Arguments:
        find_icon_option: Choice of icon-finding (bounding box finding)
         algorithm. (default: {defaults.FIND_ICON_OPTION})
        output_path: Filename for writing time and memory info to
         (default: {defaults.OUTPUT_PATH})

    Returns:
        (total time, total memory) used for find icon process
    """
    if not find_icon_option.startswith(
        "IconFinder") or find_icon_option not in globals():
      print(
          "Could not find the find icon class you inputted. Using default %s instead"
          % defaults.FIND_ICON_OPTION)
      find_icon_option = defaults.FIND_ICON_OPTION
    try:
      icon_finder = globals()[find_icon_option](self.image_list,
                                                self.icon_list)
    except KeyError:
      print("IconFinder class %s could not be resolved in global store." %
            find_icon_option)
    timer = modules.util.LatencyTimer()  # pytype: disable=module-attr
    memtracker = modules.util.MemoryTracker()  # pytype: disable=module-attr
    timer.start()
    self.proposed_boxes = icon_finder.find_icons()
    timer.stop()
    timer_info = timer.print_info(output_path)
    memtracker.run_and_track_memory(icon_finder.find_icons)
    mem_info = memtracker.print_info(output_path)
    return timer_info, mem_info

  def evaluate(self,
               visualize: bool = False,
               iou_threshold: float = defaults.IOU_THRESHOLD,
               output_path: str = defaults.OUTPUT_PATH,
               find_icon_option: str = defaults.FIND_ICON_OPTION) -> float:
    """Integrated pipeline for testing calculated bounding boxes.

    Compares calculated bounding boxes to ground truth,
    via visualization and intersection over union. Prints out accuracy
    to stdout, and also to a file via output_path.

    Arguments:
        visualize: true or false for whether to visualize
          (default: {False})
        iou_threshold: bounding boxes that yield an IOU over
         this threshold will be considered "accurate"
          (default: {defaults.IOU_THRESHOLD})
        output_path: path for where accuracy should be printed to.
        (default: {defaults.OUTPUT_PATH})
        find_icon_option: option for find_icon algorithm.
         (default: {defaults.FIND_ICON_OPTION})

    Returns:
        float -- accuracy of the bounding box detection process.
    """
    if visualize:
      self.visualize_bounding_boxes("images/gold/gold-visualized",
                                    self.gold_boxes)
      self.visualize_bounding_boxes(
          "images/" + find_icon_option + "/" + find_icon_option +
          "-visualized", self.proposed_boxes)
    ious = []
    for (proposed_box, gold_box) in zip(self.proposed_boxes, self.gold_boxes):
      ious.append(BenchmarkPipeline.calculate_iou(proposed_box, gold_box))
    accuracy = np.sum(np.array(ious) > iou_threshold) / len(ious)
    with open(output_path, "a") as output_file:
      output_file.write("Accuracy: %f\n" % accuracy)
    print("Accuracy: %f\n" % accuracy)
    return accuracy


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Run a benchmark test on find_icon algorithm.")
  parser.add_argument("--algorithm",
                      dest="find_icon_option",
                      type=str,
                      default=defaults.FIND_ICON_OPTION,
                      help="find icon algorithm option (default: %s)" %
                      defaults.FIND_ICON_OPTION)
  parser.add_argument("--tfrecord_path",
                      dest="tfrecord_path",
                      type=str,
                      default=defaults.TFRECORD_PATH,
                      help="path to tfrecord (default: %s)" %
                      defaults.TFRECORD_PATH)
  parser.add_argument(
      "--iou_threshold",
      dest="threshold",
      type=float,
      default=defaults.IOU_THRESHOLD,
      help="iou above this threshold is considered accurate (default: %f)" %
      defaults.IOU_THRESHOLD)
  parser.add_argument("--output_path",
                      dest="output_path",
                      type=str,
                      default=defaults.OUTPUT_PATH,
                      help="path to where output is written (default: %s)" %
                      defaults.OUTPUT_PATH)
  args = parser.parse_args()
  _find_icon_option = args.find_icon_option
  if _find_icon_option and (not _find_icon_option.startswith("IconFinder")
                            or _find_icon_option not in globals()):
    print(
        "Could not find the find icon class you inputted. Using default %s instead"
        % defaults.FIND_ICON_OPTION)
    _find_icon_option = defaults.FIND_ICON_OPTION
  benchmark = BenchmarkPipeline(tfrecord_path=args.tfrecord_path)
  benchmark.find_icons(find_icon_option=_find_icon_option,
                       output_path=args.output_path)
  benchmark.evaluate(iou_threshold=args.threshold,
                     output_path=args.output_path,
                     find_icon_option=_find_icon_option)
