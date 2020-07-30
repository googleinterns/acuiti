"""BenchmarkPipeline class and tfRecord utility functions."""

import argparse
from typing import List, Tuple

import cv2
import matplotlib.pyplot
from modules import defaults
from modules import icon_finder_random
from modules import util
from modules.bounding_box import BoundingBox
from modules.correctness_metrics import CorrectnessMetrics
import numpy as np

_ICON_FINDERS = {"random": icon_finder_random.IconFinderRandom}  # pytype: disable=module-attr


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

  def evaluate(
      self,
      visualize: bool = False,
      iou_threshold: float = defaults.IOU_THRESHOLD,
      output_path: str = defaults.OUTPUT_PATH,
      find_icon_option: str = defaults.FIND_ICON_OPTION,
      multi_instance_icon: bool = False
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
        find_icon_option: option for find_icon algorithm.
         (default: {defaults.FIND_ICON_OPTION})
        multi_instance_icon: flag for whether we're evaluating with
         multiple instances of an icon in an image
          (default: {False})

    Returns:
        Tuple(CorrectnessMetrics, avg runtime, avg memory of
         the bounding box detection process.)
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
      correctness = util.evaluate_proposed_bounding_boxes(
          iou_threshold, self.proposed_boxes, self.gold_boxes, output_path)
    else:
      correctness = util.evaluate_proposed_bounding_boxes(
          iou_threshold,
          [[boxes[0]] for boxes in self.proposed_boxes],
          [[boxes[0]] for boxes in self.gold_boxes],
          output_path
      )
    return correctness, avg_runtime_secs, avg_memory_mbs


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
