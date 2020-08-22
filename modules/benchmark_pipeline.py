"""BenchmarkPipeline class and tfRecord utility functions."""

import argparse
from typing import Tuple

import cv2
from modules import analysis_util
from modules import defaults
from modules import icon_finder
from modules import util
from modules.correctness_metrics import CorrectnessMetrics
from modules.types import OptionalFloat
import numpy as np


class BenchmarkPipeline:
  """Represents a pipeline to test generated Bounding Boxes.

  Usage example:
    benchmark = BenchmarkPipeline("benchmark.tfrecord")
    benchmark.evaluate()
  """

  def __init__(self, tfrecord_path: str = defaults.TFRECORD_PATH):
    # ----------------- the below are loaded from tfrecord ------------------
    parsed_image_dataset = util.parse_image_dataset(tfrecord_path)
    self.image_list, self.icon_list = util.parse_images_and_icons(
        parsed_image_dataset)  # image and template icon pairs
    self.gold_boxes = util.parse_gold_boxes(
        parsed_image_dataset)  # ground truth bounding boxes for each image

    # ----------------------the below are set by algorithm --------------------
    self.proposed_boxes = []  # proposed lists of bounding boxes for each image
    self.image_clusters = []  # list of each image's contour clusters (analysis)
    self.icon_contours = []  # list of each template icon's contours (analysis)
    self.correctness_mask = []  # True if no false pos/neg for image (analysis)

  def visualize_bounding_boxes(self,
                               output_name: str,
                               multi_instance_icon: bool = False,
                               draw_contours: bool = False,
                               only_save_failed: bool = False):
    """Visualizes bounding box of icon in its source image.

    Draws the proposed bounding boxes in red, and the gold bounding
    boxes in green. Also optionally draws the contours detected
    grouped by different colors.

    Arguments:
        output_name: prefix of filename images should be saved as
        multi_instance_icon: whether to visualize all bounding boxes
          or just the first
        draw_contours: whether to draw the contour clusters in the image
        only_save_failed: whether to save only the images that contain
         at least one false positive or false negative
    """
    for i, (image_bgr,
            icon_bgr) in enumerate(zip(self.image_list, self.icon_list)):
      gold_box_list = self.gold_boxes[i]
      proposed_box_list = self.proposed_boxes[i]
      image_bgr_copy = image_bgr.copy()
      icon_bgr_copy = icon_bgr.copy()

      # consider only the first returned icon for single-instance case
      if not multi_instance_icon:
        assert len(gold_box_list) <= 1, (
            "Length of gold box list is more than 1,",
            "but multi_instance_icon is False.")
        gold_box_list = gold_box_list[0:1]
        proposed_box_list = proposed_box_list[0:1]

      if only_save_failed and self.correctness_mask[i]:
        continue

      # draw the gold boxes in green
      for box in gold_box_list:
        # top left and bottom right corner of rectangle
        cv2.rectangle(image_bgr_copy, (box.min_x, box.min_y),
                      (box.max_x, box.max_y), (0, 255, 0), 2)

      # draw the proposed boxes in red
      for box in proposed_box_list:
        # top left and bottom right corner of rectangle
        cv2.rectangle(image_bgr_copy, (box.min_x, box.min_y),
                      (box.max_x, box.max_y), (0, 0, 255), 2)

      if draw_contours:
        # draw each contour cluster in the image with a distinct color
        # each contour cluster will alternate between these colors
        colors = [(128, 0, 128), (255, 192, 203), (255, 0, 255)]
        for j in range(0, len(self.image_clusters[i])):
          color = colors[j % len(colors)]
          cv2.drawContours(image_bgr_copy, self.image_clusters[i], j, color, 1)
        cv2.drawContours(icon_bgr_copy, [self.icon_contours[i]], -1,
                         (128, 0, 128), 1)
      image_rgb = cv2.cvtColor(image_bgr_copy, cv2.COLOR_BGR2RGB)
      icon_rgb = cv2.cvtColor(icon_bgr_copy, cv2.COLOR_BGR2RGB)
      if image_rgb is None:
        print("Could not read the image.")

      analysis_util.save_icon_with_image(icon_rgb, image_rgb,
                                         output_name + str(i) + ".png")

  def calculate_latency(self, icon_finder_object, output_path: str) -> float:
    """Uses LatencyTimer to calculate average time taken by icon_finder.

    The average time per image is via processing all the images and icons in
    the dataset. It optionally also prints this information to a file
    given by output_path. It will also record the results of the icon finder
    algorithm if no other method has done so yet.

    Arguments:
        icon_finder_object: IconFinder object that implements an
        icon-finding algorithm.
        output_path: If not empty, output will also be written
        to this path.

    Returns:
        float -- Average time in seconds that icon_finder took per image.
    """
    record_results = True
    if self.proposed_boxes:
      record_results = False

    times = []
    for image, icon in zip(self.image_list, self.icon_list):
      timer = util.LatencyTimer()  # pytype: disable=module-attr
      timer.start()
      bboxes, image_contour_clusters, icon_contour = icon_finder_object.find_icons(
          image, icon)
      timer.stop()
      if record_results:
        self.proposed_boxes.append(bboxes)
        self.image_clusters.append(image_contour_clusters)
        self.icon_contours.append(icon_contour)
      times.append(timer.calculate_latency_info(output_path))
    print("Average time per image: %f" % np.mean(times))
    return np.mean(times)

  def calculate_memory(self, icon_finder_object, output_path: str) -> float:
    """Uses MemoryTracker to calculate average memory used by icon_finder.

    The average memory is via processing all the images and icons in
    the dataset. It optionally also prints this information to a file
    given by output_path. It will also record results of the icon finder
    if no other method has recorded it yet.

    Arguments:
        icon_finder_object: IconFinder object that implements an
        icon-finding algorithm.
        output_path: If not empty, output will also be written
        to this path.

    Returns:
        float -- average memory in MiBs that icon_finder used per image.
    """
    record_results = True
    if self.proposed_boxes:
      record_results = False

    mems = []
    for image, icon in zip(self.image_list, self.icon_list):
      memtracker = util.MemoryTracker()  # pytype: disable=module-attr
      bboxes, image_contour_clusters, icon_contour = memtracker.run_and_track_memory(
          (icon_finder_object.find_icons, (image, icon)))
      mems.append(memtracker.calculate_memory_info(output_path))
      if record_results:
        self.proposed_boxes.append(bboxes)
        self.image_clusters.append(image_contour_clusters)
        self.icon_contours.append(icon_contour)
    print("Average MiBs per image: %f" % np.mean(mems))
    return np.mean(mems)

  def find_icons(
      self,
      icon_finder_object: icon_finder.IconFinder = defaults.FIND_ICON_OBJECT,
      output_path: str = defaults.OUTPUT_PATH,
      calc_latency: bool = True,
      calc_memory: bool = True) -> Tuple[OptionalFloat, OptionalFloat]:
    """Runs an icon-finding algorithm, possibly under timed and memory-tracking conditions.

    This function will ensure that the results of the icon-finding algorithm are
    recorded, either when run under timed or memory-tracking conditions, or in
    an additional iteration if neither of those have been selected. Hence, the
    maximum number of times the algorithm is run is 2 (both time and memory
    conditions selected).

    Arguments:
        icon_finder_object: IconFinder object that implements an
         icon-finding algorithm. Must be a subclass of the abstract class
         IconFinder. (default: defaults.FIND_ICON_OBJECT)
        output_path: Filename for writing time and memory info to
         (default: {defaults.OUTPUT_PATH})
        calc_latency: Whether to run algorithm under latency-profiling
         conditions. If calc_memory is True, setting calc_latency to True
         will run the algorithm an additional iteration just to calculate
         the latency information. Otherwise, the algorithm is just run once,
         recording latency information if this flag is True.
         (default: True)
        calc_memory: Whether to run algorithm under memory-profiling
         conditions. If calc_latency is True, setting calc_memory to True
         will run the algorithm an additional iteration just to calculate
         the memory information. Otherwise, the algorithm is just run once,
         recording memory information if this flag is True. (default: True)

    Returns:
        (total time, total memory) used for find icon process (floats)
         or None for each value if its corresponding boolean flag
         was passed in as False
    """
    latency = None
    memory = None
    if calc_latency:
      latency = self.calculate_latency(icon_finder_object, output_path)
    if calc_memory:
      memory = self.calculate_memory(icon_finder_object, output_path)

    # latency and memory were not run and didn't generate results
    if not self.proposed_boxes:
      for image, icon in zip(self.image_list, self.icon_list):
        bboxes, image_contour_clusters = icon_finder_object.find_icons(
            image, icon)
        self.proposed_boxes.append(bboxes)
        self.image_clusters.append(image_contour_clusters)
    return latency, memory

  def evaluate(
      self,
      visualize: bool = False,
      iou_threshold: float = defaults.IOU_THRESHOLD,
      output_path: str = defaults.OUTPUT_PATH,
      icon_finder_object: icon_finder.IconFinder = defaults.FIND_ICON_OBJECT,
      multi_instance_icon: bool = False,
      analysis_mode: bool = False,
  ) -> Tuple[CorrectnessMetrics, OptionalFloat, OptionalFloat]:
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
    if analysis_mode:
      self.image_list, self.gold_boxes = analysis_util.scale_images_and_bboxes(
          self.image_list, self.gold_boxes, 5, 5)

    avg_runtime_secs, avg_memory_mbs = self.find_icons(icon_finder_object,
                                                       output_path, True, False)
    if visualize:
      self.visualize_bounding_boxes("images/" + icon_finder_option + "/" +
                                    icon_finder_option + "-visualized",
                                    multi_instance_icon=multi_instance_icon)
    if multi_instance_icon:
      correctness, self.correctness_mask = util.evaluate_proposed_bounding_boxes(
          iou_threshold, self.proposed_boxes, self.gold_boxes, output_path)
    else:
      correctness, self.correctness_mask = util.evaluate_proposed_bounding_boxes(
          iou_threshold, [[boxes[0]] for boxes in self.proposed_boxes],
          [[boxes[0]] for boxes in self.gold_boxes], output_path)
    analysis_mode=True
    if analysis_mode:
      self.visualize_bounding_boxes("images/" + icon_finder_option +
                                    "-failed/" + icon_finder_option +
                                    "-visualized",
                                    multi_instance_icon=multi_instance_icon,
                                    draw_contours=True,
                                    only_save_failed=True)
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
