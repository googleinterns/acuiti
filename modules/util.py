"""Contains utility classes and modules, generally for the benchmark pipeline.

Utilites include:
- image processing utility functions
- evaluation utility functions
- class that measures latency
- class that measure memory usage
"""
import cProfile
import io
import pstats
from typing import Any, List, Tuple

import cv2
import memory_profiler
from modules.bounding_box import BoundingBox
from modules.confusion_matrix import ConfusionMatrix
from modules.correctness_metrics import CorrectnessMetrics
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
  """Parses each image/icon pair from each dataset entry into two corresponding lists.

  Arguments:
      parsed_image_dataset: image dataset read from TFRecord

  Returns:
      (List of images, List of icons), corresponding to image/icon pairs
      from each dataset entry.
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


def get_confusion_matrix(iou_threshold: float,
                         proposed_box_list: List[BoundingBox],
                         gold_box_list: List[BoundingBox]) -> ConfusionMatrix:
  """Count the number of true pos, true neg, false pos, false neg in proposed boxes.

  Overview of algorithm: find the gold box ("match") that maximizes IOU for each
  proposed box. If two proposed boxes match to the same gold box, use the higher
  IOU one (the other proposed box is a false positive). If there are no gold
  boxes available, each proposed box is a false positive. Then, for each match
  where the IOU is above the IOU threshold, we have a true positive. Otherwise,
  each match is counted as one false positive and one false negative.
  Finally, if there are no gold boxes and no proposed boxes we have a true
  negative.

  Arguments:
      iou_threshold: a proposed box that has an IOU with a gold box below this
       threshold is effectively a distinct and separate box.
      proposed_box_list: a list of BoundingBoxes proposed for an image
      gold_box_list: a list of ground truth BoundingBoxes for the image

  Returns:
      ConfusionMatrix of false pos/neg and true pos/neg
  """
  num_false_pos = 0  # fp
  num_false_neg = 0  # fn
  num_true_pos = 0  # tp
  num_true_neg = 0  # tn

  # mapping from gold boxes to the IOU of their best matching proposed box
  # keys: index of gold box that maximizes the iou of a given proposed box
  #   (not all gold box indices will necessarily be matched to a proposed box
  #   and placed in the dict; len(matched...) = the number of gold boxes that
  #   did such receive a match)
  # values: corresponding iou between the proposed box and gold box
  #   (if two proposed boxes 'match' to the same gold box,
  #   lower iou proposed box is discarded)
  matched_gold_index_to_proposed_iou = {}
  num_proposed_box_without_match = 0
  for proposed_box in proposed_box_list:
    # find the index of the gold box that maximizes IOU w/a given proposed box
    max_iou = -1
    max_gold_box_index = -1
    for gold_index, gold_box in enumerate(gold_box_list):
      iou = proposed_box.calculate_iou(gold_box)
      if iou > max_iou:
        max_iou = iou
        max_gold_box_index = gold_index

    # check if there were no gold boxes to begin with
    if max_gold_box_index == -1:
      num_proposed_box_without_match += 1

    # if the proposed box matched to a gold box that another
    # proposed box already matched with, discard the one with lower IOU
    # and update the IOU
    elif max_gold_box_index in matched_gold_index_to_proposed_iou:
      num_proposed_box_without_match += 1  # a proposed box is discarded
      if matched_gold_index_to_proposed_iou[max_gold_box_index] < max_iou:
        matched_gold_index_to_proposed_iou[max_gold_box_index] = max_iou

    # finally, if no other proposed box has matched to this gold box,
    # place it in our dictionary along with the iou
    else:
      matched_gold_index_to_proposed_iou[max_gold_box_index] = max_iou

  # true positives are matches that meet the IOU threhsold
  current_true_pos = np.sum(
      np.array(list(matched_gold_index_to_proposed_iou.values())) >=
      iou_threshold)
  num_true_pos += current_true_pos

  # matches w/IOU below threshold: corresp. proposed boxes are false positives
  num_false_pos += len(matched_gold_index_to_proposed_iou) - current_true_pos
  # matches w/IOU below threshold: corresp. gold boxes are false negatives
  num_false_neg += len(matched_gold_index_to_proposed_iou) - current_true_pos

  # proposed boxes that didn't get matched to a gold box, because none exists,
  # or another proposed box matched to the gold box with a higher IOU,
  # are false positives
  num_false_pos += num_proposed_box_without_match

  # gold boxes that were not matched at all are false negatives
  num_false_neg += len(gold_box_list) - len(matched_gold_index_to_proposed_iou)
  if not proposed_box_list and not gold_box_list:
    num_true_neg += 1  # correctly identified that icon didn't appear in image

  return ConfusionMatrix(num_false_pos, num_false_neg, num_true_pos,
                         num_true_neg)


def evaluate_proposed_bounding_boxes(
    iou_threshold: float,
    proposed_boxes: List[List[BoundingBox]],
    gold_boxes: List[List[BoundingBox]],
    output_path: str = defaults.OUTPUT_PATH,
) -> Tuple[CorrectnessMetrics, List[bool]]:
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
      Tuple(CorrectnessMetrics, correctness mask corresponding
       to whether we had zero false pos/neg for each image)
  """
  total_confusion_matrix = ConfusionMatrix(0, 0, 0, 0)
  correctness_mask = [False] * len(gold_boxes)
  for i, (proposed_box_list,
          gold_box_list) in enumerate(zip(proposed_boxes, gold_boxes)):
    confusion_matrix = get_confusion_matrix(iou_threshold, proposed_box_list,
                                            gold_box_list)
    correctness_mask[i] = (confusion_matrix.false_pos +
                           confusion_matrix.false_neg == 0)
    total_confusion_matrix += confusion_matrix
  correctness_metrics = total_confusion_matrix.calculate_correctness_metrics(
      output_path)
  return correctness_metrics, correctness_mask


class LatencyTimer:
  """Wrapper class for cython runtime profiler.
  """

  def __init__(self):
    self.pr = cProfile.Profile()

  def start(self):
    self.pr.enable()

  def stop(self):
    self.pr.disable()

  def calculate_latency_info(self,
                             output_path: str = defaults.OUTPUT_PATH) -> float:
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
    # parse cProfiler's output to get the total time as a float
    first_line = info.partition("\n")[0]
    total_time = first_line.split(" ")[-2]
    return float(total_time)


class MemoryTracker:
  """Wrapper class for PyPI's memory-profiler.
  """

  def __init__(self):
    self.memory_info = ""
    self.prev_memory = 0

  def run_and_track_memory(self, func_args_tuple) -> Any:
    """Tracks memory usage of a function.

    Args:
        func_args_tuple: tuple of function and
         variable number of arguments to the function
          in the form of (f, args, kw)
          Example argument: (f, (1,), {'n': int(1e6)})
    Returns:
        Returns whatever the function returns (could be None)
    """
    self.prev_memory = memory_profiler.memory_usage()[-1]
    self.memory_info, return_value = memory_profiler.memory_usage(
        func_args_tuple, retval=True)
    return return_value

  def calculate_memory_info(self,
                            output_path: str = defaults.OUTPUT_PATH) -> float:
    """Calculates memory usage info and optionally prints to file.

    Args:
        output_path: file path to print output to.
         (default: None)

    Returns:
        float -- the MiBs used by the function call
    """

    mb_used = np.max(self.memory_info) - self.prev_memory
    output_msg = "Process took %f MiBs \n" % mb_used
    if output_path:
      with open(output_path, "a") as output_file:
        output_file.write(output_msg)
    return float(mb_used)
