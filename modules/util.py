"""Contains utility classes.

So far, we have classes that measure latency and
memory usage, image processing utility functions,
and evaluation functions.
"""
import cProfile
import io
import pstats
from typing import Any, List, Tuple

import cv2
import memory_profiler
from modules.bounding_box import BoundingBox
import modules.defaults as defaults
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


def _parse_image_function(
    example_proto: tf.python.framework.ops.Tensor) -> Any:
  return tf.io.parse_single_example(example_proto, _IMAGE_FEATURE_DESCRIPTION)


def parse_image_dataset(
    path: str) -> tf.python.data.ops.dataset_ops.MapDataset:
  raw_dataset = tf.data.TFRecordDataset(path)
  parsed_image_dataset = raw_dataset.map(_parse_image_function)
  return parsed_image_dataset


def parse_gold_boxes(
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


def parse_images_and_icons(
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


def _count_true_false_pos_neg(
    iou_threshold: float, proposed_boxes: List[List[BoundingBox]],
    gold_boxes: List[List[BoundingBox]]) -> Tuple[float, float, float, float]:
  """Count the number of true pos, true neg, false pos, false neg in proposed boxes.

  Arguments:
      iou_threshold: a proposed box that has an IOU with a gold box below this
       threshold is effectively a distinct and separate box.
      proposed_boxes: a list of BoundingBox lists, one for each image,
       as the proposed box.
      gold_boxes: a list of BoundingBox lists, one for each image,
       as the ground truth.

  Returns:
      Tuple(Number of false pos, false neg, true pos, true neg)
  """
  num_false_pos = 0
  num_false_neg = 0
  num_true_pos = 0
  num_true_neg = 0
  for proposed_box_list, gold_box_list in zip(proposed_boxes, gold_boxes):
    matches = {}
    for proposed_box in proposed_box_list:
      # find the gold box that maximizes IOU
      max_iou = 0
      max_gold_box = None
      for gold_box in gold_box_list:
        iou = proposed_box.calculate_iou(gold_box)
        if iou > max_iou:
          max_iou = iou
          max_gold_box = gold_box
      # if the gold box already has a match, increase false positives
      # (condition holds even if there are no gold boxes; max_gold_box = None)
      proposed_iou = max_iou
      if max_gold_box in matches:
        num_false_pos += 1
        current_max_iou = matches[max_gold_box]
        if current_max_iou > proposed_iou:
          proposed_iou = matches[max_gold_box]
      matches[max_gold_box] = proposed_iou
    # evaluate matches with double penalty for mismatches
    current_true_pos = np.sum(
        [1 if iou >= iou_threshold else 0 for iou in matches.values()])
    num_true_pos += current_true_pos
    num_false_pos += len(matches) - current_true_pos
    num_false_neg += len(matches) - current_true_pos
    num_false_neg += len(gold_box_list) - len(matches)
    if not proposed_box_list and not gold_box_list:
      num_true_neg += 1
  return num_false_pos, num_false_neg, num_true_pos, num_true_neg


def evaluate_proposed_bounding_boxes(
    iou_threshold: float,
    proposed_boxes: List[List[BoundingBox]],
    gold_boxes: List[List[BoundingBox]],
    output_path: str = defaults.OUTPUT_PATH,
) -> Tuple[float, float, float]:
  """Evaluates proposed boxes against gold boxes.

  Arguments:
      iou_threshold: a proposed box that has an IOU with a gold box below this
       threshold is effectively a distinct and separate box.
      proposed_boxes: a list of BoundingBox lists, one for each image,
       as the proposed box.
      gold_boxes: a list of BoundingBox lists, one for each image,
       as the ground truth.
      output_path: if not None, prints accuracy, precision, and recall
       to file at path.

  Returns:
      Tuple(accuracy, precision, recall)
  """
  num_false_pos, num_false_neg, num_true_pos, num_true_neg = _count_true_false_pos_neg(
      iou_threshold, proposed_boxes, gold_boxes)
  accuracy = (num_true_pos + num_true_neg) / (num_true_pos + num_true_neg +
                                              num_false_pos + num_false_neg)
  if num_true_pos == 0 and num_false_pos == 0:
    precision = 1
  else:
    precision = num_true_pos / (num_true_pos + num_false_pos)
  if num_true_pos == 0 and num_false_neg == 0:
    recall = 1
  else:
    recall = num_true_pos / (num_true_pos + num_false_neg)

  if output_path:
    with open(output_path, "a") as output_file:
      output_file.write("Accuracy: %f\n" % accuracy)
      output_file.write("Precision: %f\n" % precision)
      output_file.write("Recall: %f\n" % recall)

  print("Accuracy: %f\n" % accuracy)
  print("Precision: %f\n" % precision)
  print("Recall: %f\n" % recall)
  return accuracy, precision, recall


class LatencyTimer:
  """Wrapper class for cython runtime profiler.
  """

  def __init__(self):
    self.pr = cProfile.Profile()

  def start(self):
    self.pr.enable()

  def stop(self):
    self.pr.disable()

  def calculate_info(self, output_path=defaults.OUTPUT_PATH):
    """Calculates latency info and optionally prints to file.

    Args:
        output_path: file path to print output to.
         (default: None)

    Returns:
        float -- the total seconds taken between calls to start and stop
    """
    s = io.StringIO()
    sort_by = pstats.SortKey.CUMULATIVE  # pytype: disable=module-attr
    ps = pstats.Stats(self.pr, stream=s).sort_stats(sort_by)
    ps.print_stats()
    info = s.getvalue()
    if output_path:
      with open(output_path, "a") as output_file:
        output_file.write(info)
    print(info)
    # parse cProfiler's output to get the total time as a float
    first_line = info.partition("\n")[0]
    total_time = first_line.split(" ")[-2]
    return float(total_time)


class MemoryTracker:
  """Wrapper class for PyPI's memory-profiler.
  """

  def __init__(self):
    self.memory_info = ""

  def run_and_track_memory(self, func_args_tuple):
    """Tracks memory usage of a function.

    Args:
        func_args_tuple: tuple of function and
         variable number of arguments to the function
          in the form of (f, args, kw)
          Example argument: (f, (1,), {'n': int(1e6)})
    """
    self.memory_info = memory_profiler.memory_usage(func_args_tuple)

  def calculate_info(self, output_path=defaults.OUTPUT_PATH):
    """Calculates memory usage info and optionally prints to file.

    Args:
        output_path: file path to print output to.
         (default: None)

    Returns:
        float -- the MiBs used by the function call
    """
    average_mb = self.memory_info
    if len(self.memory_info) > 1:
      average_mb = np.mean(self.memory_info)
    output_msg = "Process took %f MiBs \n" % average_mb
    if output_path:
      with open(output_path, "a") as output_file:
        output_file.write(output_msg)
    print(output_msg)
    return average_mb
