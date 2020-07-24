"""BenchmarkPipeline class and tfRecord utility functions."""

import argparse
from typing import Any, List, Tuple

import cv2
import matplotlib.pyplot
from modules import defaults
from modules import icon_finder_random
from modules import util
from modules.bounding_box import BoundingBox
import numpy as np
import tensorflow as tf

_IMAGE_FEATURE_DESCRIPTION = {
    "encoded_image_png": tf.io.FixedLenFeature([], tf.string),
    "encoded_icon_png": tf.io.FixedLenFeature([], tf.string),
    "box_ymin": tf.io.FixedLenSequenceFeature([],
                                              tf.float32,
                                              allow_missing=True),
    "box_xmin": tf.io.FixedLenSequenceFeature([],
                                              tf.float32,
                                              allow_missing=True),
    "box_ymax": tf.io.FixedLenSequenceFeature([],
                                              tf.float32,
                                              allow_missing=True),
    "box_xmax": tf.io.FixedLenSequenceFeature([],
                                              tf.float32,
                                              allow_missing=True),
}

_ICON_FINDERS = {"random": icon_finder_random.IconFinderRandom}  # pytype: disable=module-attr


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
) -> List[List[BoundingBox]]:
  """Retrieve a list of bounding boxes from dataset.

  Arguments:
      parsed_image_dataset: Original dataset read from TFRecord

  Returns:
      List of lists of ground truth BoundingBoxes.
  """
  all_images_gold_boxes = []
  for image_features in parsed_image_dataset:
    single_image_gold_boxes = []
    for xmin, ymin, xmax, ymax in zip(image_features["box_xmin"],
                                      image_features["box_ymin"],
                                      image_features["box_xmax"],
                                      image_features["box_ymax"]):
      gold_box = BoundingBox(xmin, ymin, xmax, ymax)
      single_image_gold_boxes.append(gold_box)
    all_images_gold_boxes.append(single_image_gold_boxes)
  return all_images_gold_boxes


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
    benchmark.evaluate()
  """

  def __init__(self, tfrecord_path: str = defaults.TFRECORD_PATH):
    parsed_image_dataset = _parse_image_dataset(tfrecord_path)
    self.gold_boxes = _parse_gold_boxes(parsed_image_dataset)
    self.image_list, self.icon_list = _parse_images_and_icons(
        parsed_image_dataset)
    self.proposed_boxes = []

  def visualize_bounding_boxes(self, output_name: str,
                               boxes: List[BoundingBox]):
    """Visualizes bounding box of icon in its source image.

    Arguments:
        output_name: prefix of filename images should be saved as
        boxes: list of BoundingBoxes
    """
    for i, image_bgr in enumerate(self.image_list):
      box_list = boxes[i]
      for box in box_list:
        # top left and bottom right corner of rectangle
        cv2.rectangle(image_bgr, (box.min_x, box.min_y),
                      (box.max_x, box.max_y), (0, 255, 0), 3)
      image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

      if image_rgb is None:
        print("Could not read the image.")
      matplotlib.pyplot.imshow(image_rgb)
      matplotlib.pyplot.imsave(output_name + str(i) + ".png", image_rgb)

  def calculate_latency(self, icon_finder, output_path: str) -> float:
    """Uses LatencyTimer to calculate average time taken by icon_finder.

    The average time per image is via processing all the images and icons in
    the dataset. It optionally also prints this information to a file
    given by output_path.

    Arguments:
        icon_finder: IconFinder object that implements an
        icon-finding algorithm.
        output_path: If not empty, output will also be written
        to this path.

    Returns:
        float -- Average time in seconds that icon_finder took per image.
    """
    times = []
    for image, icon in zip(self.image_list, self.icon_list):
      timer = util.LatencyTimer()  # pytype: disable=module-attr
      timer.start()
      self.proposed_boxes.append(icon_finder.find_icons(image, icon))
      timer.stop()
      times.append(timer.calculate_info(output_path))
    return np.mean(times)

  def calculate_memory(self, icon_finder, output_path: str) -> float:
    """Uses MemoryTracker to calculate average memory used by icon_finder.

    The average memory is via processing all the images and icons in
    the dataset. It optionally also prints this information to a file
    given by output_path.

    Arguments:
        icon_finder: IconFinder object that implements an
        icon-finding algorithm.
        output_path: If not empty, output will also be written
        to this path.

    Returns:
        float -- average memory in MiBs that icon_finder used per image.
    """
    mems = []
    for image, icon in zip(self.image_list, self.icon_list):
      memtracker = util.MemoryTracker()  # pytype: disable=module-attr
      memtracker.run_and_track_memory((icon_finder.find_icons, (image, icon)))
      mems.append(memtracker.calculate_info(output_path))
    return np.mean(mems)

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
    if find_icon_option not in _ICON_FINDERS:
      print(
          "Could not find the find icon class you inputted. Using default %s instead"
          % defaults.FIND_ICON_OPTION)
      find_icon_option = defaults.FIND_ICON_OPTION
    try:
      icon_finder = _ICON_FINDERS[find_icon_option]()
    except KeyError:
      print("Error resolving %s" % _ICON_FINDERS[find_icon_option])

    return self.calculate_latency(icon_finder,
                                  output_path), self.calculate_memory(
                                      icon_finder, output_path)

  def single_instance_eval(self, iou_threshold: float,
                           output_path: str) -> float:
    """Evaluates proposed bounding boxes with one instance of icon in image.

    Arguments:
        iou_threshold: Threshold above which a bbox is accurate
        output_path: Output path of writing accuracy info

    Returns:
        float -- accuracy calculated
    """
    ious = []
    for (proposed_box_list, gold_box_list) in zip(self.proposed_boxes,
                                                  self.gold_boxes):
      ious.append(proposed_box_list[0].calculate_iou(gold_box_list[0]))
    accuracy = np.sum(np.array(ious) > iou_threshold) / len(ious)
    if output_path:
      with open(output_path, "a") as output_file:
        output_file.write("Accuracy: %f\n" % accuracy)
    print("Accuracy: %f\n" % accuracy)
    return accuracy

  def multi_instance_eval(self, iou_threshold: float,
                          output_path: str) -> float:
    raise NotImplementedError

  def evaluate(
      self,
      visualize: bool = False,
      iou_threshold: float = defaults.IOU_THRESHOLD,
      output_path: str = defaults.OUTPUT_PATH,
      find_icon_option: str = defaults.FIND_ICON_OPTION,
      multi_instance_icon: bool = False) -> Tuple[float, float, float]:
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
        multi_instance_icon: flag for whether we're evaluating with
         multiple instances of an icon in an image
          (default: {False})

    Returns:
        float, float, float -- accuracy, avg runtime, avg memory of the bounding
         box detection process.
    """
    avg_runtime_secs, avg_memory_mbs = self.find_icons(find_icon_option,
                                                       output_path)
    if visualize:
      self.visualize_bounding_boxes("images/gold/gold-visualized",
                                    self.gold_boxes)
      self.visualize_bounding_boxes(
          "images/" + find_icon_option + "/" + find_icon_option +
          "-visualized", self.proposed_boxes)
    if multi_instance_icon:
      accuracy = self.multi_instance_eval()
    else:
      accuracy = self.single_instance_eval(iou_threshold, output_path)
    return accuracy, avg_runtime_secs, avg_memory_mbs


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
  parser.add_argument(
      "--multi_instance_icon",
      dest="multi_instance_icon",
      type=bool,
      default=False,
      help=
      "whether to evaluate with multiple instances of an icon in an image (default: %s)"
      % False)
  parser.add_argument(
      "--visualize",
      dest="visualize",
      type=bool,
      default=False,
      help="whether to visualize bounding boxes on image (default: %s)" %
      False)
  args = parser.parse_args()
  _find_icon_option = args.find_icon_option
  if _find_icon_option not in _ICON_FINDERS:
    print(
        "Could not find the find icon class you inputted. Using default %s instead"
        % defaults.FIND_ICON_OPTION)
    _find_icon_option = defaults.FIND_ICON_OPTION
  benchmark = BenchmarkPipeline(tfrecord_path=args.tfrecord_path)
  benchmark.evaluate(visualize=args.visualize,
                     iou_threshold=args.threshold,
                     output_path=args.output_path,
                     find_icon_option=_find_icon_option,
                     multi_instance_icon=args.multi_instance_icon)
