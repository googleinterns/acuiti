"""BenchmarkPipeline class and tfRecord utility functions."""

import os
import sys

import argparse
from bounding_box import BoundingBox
from bounding_box_generator import BBGenerator
import cv2
from defaults import DEFAULT_ARGS
# from ..directory import ROOT_PATH
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from util import LatencyTimer
from util import MemoryTracker

image_feature_description = {
    "encoded_image_png": tf.io.FixedLenFeature([], tf.string),
    "encoded_icon_png": tf.io.FixedLenFeature([], tf.string),
    "box_ymin": tf.io.FixedLenFeature([], tf.float32),
    "box_xmin": tf.io.FixedLenFeature([], tf.float32),
    "box_ymax": tf.io.FixedLenFeature([], tf.float32),
    "box_xmax": tf.io.FixedLenFeature([], tf.float32),
}


def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, image_feature_description)


def _parse_image_dataset(path):
  raw_dataset = tf.data.TFRecordDataset(path)
  parsed_image_dataset = raw_dataset.map(_parse_image_function)
  return parsed_image_dataset


def _parse_bb_gold(parsed_image_dataset):
  bb_gold_list = []
  for image_features in parsed_image_dataset:
    bb_gold = BoundingBox(image_features["box_xmin"],
                          image_features["box_ymin"],
                          image_features["box_xmax"],
                          image_features["box_ymax"])
    bb_gold_list.append(bb_gold)
  return bb_gold_list


def _parse_images_and_icons(parsed_image_dataset):
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
    benchmark = BenchmarkPipeline("acuiti/benchmark.tfrecord")
    benchmark.find_icons()
    benchmark.evaluate()
  """
  def __init__(self, tfrecord_path=DEFAULT_ARGS["tfrecord_path"]):
    self.parsed_image_dataset = _parse_image_dataset(tfrecord_path)
    self.bb_gold_list = _parse_bb_gold(self.parsed_image_dataset)
    self.image_list, self.icon_list = _parse_images_and_icons(
        self.parsed_image_dataset)
    self.bb_list = []

  def visualize_bounding_boxes(self, output_name, bb_list):
    """Visualizes bounding box of icon in its source image.

    Arguments:
        output_name {string} -- prefix of filename images should be saved as
        bb_list {[BoundingBox]} -- list of BoundingBoxes
    """
    for i, image_features in enumerate(self.parsed_image_dataset):
      image_raw = image_features["encoded_image_png"].numpy()
      image_bgr = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), -1)
      bb = bb_list[i]
      # top left and bottom right corner of rectangle
      cv2.rectangle(image_bgr, (bb.min_x, bb.min_y), (bb.max_x, bb.max_y),
                    (0, 255, 0), 3)
      image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

      if image_rgb is None:
        print("Could not read the image.")
      pyplot.imshow(image_rgb)
      pyplot.imsave(output_name + str(i) + ".jpg", image_rgb)

  @staticmethod
  def calculate_iou(bb, bb_gold):
    """Calculate the intersection over union of two bounding boxes.

    The intersection is the overlap of two bounding boxes,
    and the union is the total area of two bounding boxes.

    Arguments:
      bb {BoundingBox} -- calculated BoundingBox.
      bb_gold {BoundingBox} -- ground truth BoundingBox.

    Returns:
      float -- intersection over union of the two bounding boxes.
    """
    overlap_box = BoundingBox(max(bb.min_x, bb_gold.min_x),
                              max(bb.min_y, bb_gold.min_y),
                              min(bb.max_x, bb_gold.max_x),
                              min(bb.max_y, bb_gold.max_y))

    if overlap_box.max_x < overlap_box.min_x or overlap_box.max_y < overlap_box.min_y:
      return 0.0

    intersection_area = overlap_box.calculate_area()
    bb_area = bb.calculate_area()
    bb_gold_area = bb_gold.calculate_area()
    iou = intersection_area / float(bb_area + bb_gold_area - intersection_area)
    return iou

  def find_icons(self,
                 generator_option=DEFAULT_ARGS["generator_option"],
                 output_path=DEFAULT_ARGS["output_path"]):
    """Runs an icon-finding algorithm under timed and memory-tracking conditions.

    Keyword Arguments:
        generator_option {str} -- Choice of icon-finding (bounding box finding)
         algorithm. (default: {"random"})

    Returns:
      float, float -- total time and total memory used for find icon process
    """
    bb_generator = BBGenerator(self.image_list, self.icon_list)
    timer = LatencyTimer()
    memtracker = MemoryTracker()
    timer.start()
    if generator_option is "random":
      self.bb_list = bb_generator.generate_random()
    timer.stop()
    timer_info = timer.print_info(output_path)
    memtracker.run_and_track_memory(bb_generator.generate_random)
    mem_info = memtracker.print_info(output_path)
    return timer_info, mem_info

  def evaluate(self,
               iou_threshold=DEFAULT_ARGS["iou_threshold"],
               output_path=DEFAULT_ARGS["output_path"],
               generator_option=DEFAULT_ARGS["generator_option"]):
    """Integrated pipeline for testing calculated bounding boxes.

    Compares calculated bounding boxes to ground truth,
    via visualization and intersection over union.

    Keyword Arguments:
        iou_threshold {float} -- bounding boxes that yield an IOU over
         this threshold will be considered "accurate" (default: {0.6})

    Returns:
      float -- accuracy of the bounding box detection process.
    """
    self.visualize_bounding_boxes("images/gold/gold-visualized",
                                  self.bb_gold_list)
    self.visualize_bounding_boxes(
        "images/" + generator_option + "/" + generator_option + "-visualized",
        self.bb_list)
    ious = []
    for (bb, bb_gold) in zip(self.bb_list, self.bb_gold_list):
      ious.append(BenchmarkPipeline.calculate_iou(bb, bb_gold))
    accuracy = str(np.sum(np.array(ious) > iou_threshold) / len(ious))
    output_file = open(output_path, "a")
    output_file.write("Accuracy: " + accuracy + "\n")
    output_file.close()
    print("Accuracy: " + accuracy + "\n")
    return accuracy


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Run a benchmark test on find_icon algorithm.")
  parser.add_argument("--algorithm",
                      dest="generator_option",
                      type=str,
                      default=DEFAULT_ARGS["generator_option"],
                      help="find icon algorithm option (default: random)")
  parser.add_argument("--tfrecord_path",
                      dest="tfrecord_path",
                      type=str,
                      default=DEFAULT_ARGS["tfrecord_path"],
                      help="path to tfrecord (default: benchmark.tfrecord)")
  parser.add_argument(
      "--iou_threshold",
      dest="threshold",
      type=float,
      default=DEFAULT_ARGS["iou_threshold"],
      help="iou above this threshold is considered accurate (default: 0.6)")
  parser.add_argument(
      "--output_path",
      dest="output_path",
      type=str,
      default=DEFAULT_ARGS["output_path"],
      help="path to where output is written (default: out.txt)")
  args = parser.parse_args()
  benchmark = BenchmarkPipeline(tfrecord_path=args.tfrecord_path)
  benchmark.find_icons(generator_option=args.generator_option,
                       output_path=args.output_path)
  benchmark.evaluate(iou_threshold=args.threshold,
                     output_path=args.output_path,
                     generator_option=args.generator_option)
