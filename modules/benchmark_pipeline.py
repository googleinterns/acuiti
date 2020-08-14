"""BenchmarkPipeline class and tfRecord utility functions."""

import argparse
from typing import Tuple

import cv2
import matplotlib.pyplot
from modules import analysis_util
from modules import defaults
from modules import icon_finder
from modules import util
from modules.correctness_metrics import CorrectnessMetrics
import numpy as np


class BenchmarkPipeline:
  """Represents a pipeline to test generated Bounding Boxes.

  Usage example:
    benchmark = BenchmarkPipeline("benchmark.tfrecord")
    benchmark.evaluate()
  """

  def __init__(self, tfrecord_path: str = defaults.TFRECORD_PATH):
    parsed_image_dataset = util.parse_image_dataset(tfrecord_path)
    self.gold_boxes = util.parse_gold_boxes(parsed_image_dataset)
    self.image_list, self.icon_list = util.parse_images_and_icons(
        parsed_image_dataset)
    self.proposed_boxes = []
    self.image_clusters = []

  def visualize_bounding_boxes(self,
                               output_name: str,
                               multi_instance_icon: bool = False,
                               draw_contours: bool = True):
    """Visualizes bounding box of icon in its source image.

    Draws the proposed bounding boxes in red, and the gold bounding
    boxes in green. Also optionally draws the contours detected
    grouped by different colors.

    Arguments:
        output_name: prefix of filename images should be saved as
        multi_instance_icon: whether to visualize all bounding boxes
          or just the first
        draw_contours: whether to draw the contour clusters in the image
    """
    for i, image_bgr in enumerate(self.image_list):
      gold_box_list = self.gold_boxes[i]
      proposed_box_list = self.proposed_boxes[i]
      image_bgr_copy = image_bgr.copy()

      # consider only the first returned icon for single-instance case
      if not multi_instance_icon:
        assert len(gold_box_list) <= 1, (
            "Length of gold box list is more than 1,",
            "but multi_instance_icon is False.")
        gold_box_list = gold_box_list[0:1]
        proposed_box_list = proposed_box_list[0:1]

      # draw the gold boxes in green
      for box in gold_box_list:
        # top left and bottom right corner of rectangle
        cv2.rectangle(image_bgr_copy, (box.min_x, box.min_y),
                      (box.max_x, box.max_y), (0, 255, 0), 3)

      # draw the proposed boxes in red
      for box in proposed_box_list:
        # top left and bottom right corner of rectangle
        cv2.rectangle(image_bgr_copy, (box.min_x, box.min_y),
                      (box.max_x, box.max_y), (0, 0, 255), 3)

      if draw_contours:
        # draw each contour cluster in the image with a distinct color
        # each contour cluster will alternate between these colors
        colors = [(128, 0, 128), (255, 192, 203), (255, 0, 255)]
        for j in range(0, len(self.image_clusters[i])):
          color = colors[j % len(colors)]
          cv2.drawContours(image_bgr_copy, self.image_clusters[i], j, color, 1)
      image_rgb = cv2.cvtColor(image_bgr_copy, cv2.COLOR_BGR2RGB)

      if image_rgb is None:
        print("Could not read the image.")
      matplotlib.pyplot.imshow(image_rgb)
      matplotlib.pyplot.imsave(output_name + str(i) + ".png", image_rgb)

  def calculate_latency(self, icon_finder_object, output_path: str) -> float:
    """Uses LatencyTimer to calculate average time taken by icon_finder.

    The average time per image is via processing all the images and icons in
    the dataset. It optionally also prints this information to a file
    given by output_path.

    Arguments:
        icon_finder_object: IconFinder object that implements an
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
      _, _ = icon_finder_object.find_icons(image, icon)
      timer.stop()
      times.append(timer.calculate_info(output_path))
    print("Average time per image: %f" % np.mean(times))
    return np.mean(times)

  def calculate_memory(self, icon_finder_object, output_path: str) -> float:
    """Uses MemoryTracker to calculate average memory used by icon_finder.

    The average memory is via processing all the images and icons in
    the dataset. It optionally also prints this information to a file
    given by output_path.

    Arguments:
        icon_finder_object: IconFinder object that implements an
        icon-finding algorithm.
        output_path: If not empty, output will also be written
        to this path.

    Returns:
        float -- average memory in MiBs that icon_finder used per image.
    """
    mems = []
    for image, icon in zip(self.image_list, self.icon_list):
      memtracker = util.MemoryTracker()  # pytype: disable=module-attr
      memtracker.run_and_track_memory(
          (icon_finder_object.find_icons, (image, icon)))
      mems.append(memtracker.calculate_info(output_path))
    print("Average MiBs per image: %f" % np.mean(mems))
    return np.mean(mems)

  def find_icons(
      self,
      icon_finder_object: icon_finder.IconFinder = defaults.FIND_ICON_OBJECT,
      output_path: str = defaults.OUTPUT_PATH,
      calc_latency: bool = True,
      calc_memory: bool = True) -> Tuple[float, float]:
    """Runs an icon-finding algorithm under timed and memory-tracking conditions.

    Arguments:
        icon_finder_object: IconFinder object that implements an
         icon-finding algorithm. Must be a subclass of the abstract class
         IconFinder. (default: defaults.FIND_ICON_OBJECT)
        output_path: Filename for writing time and memory info to
         (default: {defaults.OUTPUT_PATH})
        calc_latency: Whether to run algorithm under latency-profiling
         conditions (default: True)
        calc_memory: Whether to run algorithm under memory-profiling
         conditions (default: True)

    Returns:
        (total time, total memory) used for find icon process
         or -1 for each value if its corresponding boolean flag
         was passed in as False
    """
    for image, icon in zip(self.image_list, self.icon_list):
      bboxes, image_contour_clusters = icon_finder_object.find_icons(
          image, icon)
      self.proposed_boxes.append(bboxes)
      self.image_clusters.append(image_contour_clusters)

    latency = -1
    memory = -1
    if calc_latency:
      latency = self.calculate_latency(icon_finder_object, output_path)
    if calc_memory:
      memory = self.calculate_memory(icon_finder_object, output_path)
    return latency, memory

  def evaluate(
      self,
      visualize: bool = False,
      iou_threshold: float = defaults.IOU_THRESHOLD,
      output_path: str = defaults.OUTPUT_PATH,
      icon_finder_object: icon_finder.IconFinder = defaults.FIND_ICON_OBJECT,
      multi_instance_icon: bool = False,
      analysis_mode: bool = False,
  ) -> Tuple[CorrectnessMetrics, float, float]:
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
        icon_finder_object: option for find_icon algorithm.
         (default: {defaults.FIND_ICON_OBJECT})
        multi_instance_icon: flag for whether we're evaluating with
         multiple instances of an icon in an image
          (default: {False})
        analysis_mode: bool for whether to run extra analyses, similar
         to debug mode.

    Returns:
        Tuple(CorrectnessMetrics, avg runtime, avg memory of
         the bounding box detection process.)
    """
    assert isinstance(
        icon_finder_object, icon_finder.IconFinder
    ), "Icon-finding object passed in must be an instance of IconFinder"
    icon_finder_option = type(icon_finder_object).__name__
    avg_runtime_secs, avg_memory_mbs = self.find_icons(icon_finder_object,
                                                       output_path, True, True)
    if visualize:
      self.visualize_bounding_boxes("images/" + icon_finder_option + "/" +
                                    icon_finder_option + "-visualized",
                                    multi_instance_icon=multi_instance_icon)
    if multi_instance_icon:
      correctness = util.evaluate_proposed_bounding_boxes(
          iou_threshold, self.proposed_boxes, self.gold_boxes, output_path)
    else:
      correctness = util.evaluate_proposed_bounding_boxes(
          iou_threshold, [[boxes[0]] for boxes in self.proposed_boxes],
          [[boxes[0]] for boxes in self.gold_boxes], output_path)
    if analysis_mode:
      analysis_util.label_cluster_size(self.image_clusters, self.image_list,
                                       "images/labelled-contours/")
      samples = []
      for clusters in self.image_clusters:
        samples.extend(map(len, clusters))
      title = "Number of keypoints in image clusters"
      analysis_util.generate_histogram(np.array(samples), title, title,
                                       "keypoints-histogram.png")
    return correctness, avg_runtime_secs, avg_memory_mbs


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Run a benchmark test on find_icon algorithm.")
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

  benchmark = BenchmarkPipeline(tfrecord_path=args.tfrecord_path)
  benchmark.evaluate(visualize=args.visualize,
                     iou_threshold=args.threshold,
                     output_path=args.output_path,
                     multi_instance_icon=args.multi_instance_icon)
