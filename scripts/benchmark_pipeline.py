"""BenchmarkPipeline class and tfRecord utility functions."""

import time
from bb import BoundingBox
from bb_generator import BBGenerator
import cv2
from matplotlib import pyplot
import numpy as np
import tensorflow as tf

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


def _parse_images(path):
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


class BenchmarkPipeline:
  """Represents a pipeline to test generated Bounding Boxes.

  Usage example:
    pipeline = BenchmarkPipeline("acuiti/benchmark.tfrecord")
    pipeline.integrated_pipeline()
  """

  def __init__(self, tfrecord_path):
    self.parsed_image_dataset = _parse_images(tfrecord_path)
    self.bb_gold_list = _parse_bb_gold(self.parsed_image_dataset)

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

  def calculate_iou(self, bb, bb_gold):
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

  def integrated_pipeline(self):
    """Integrated pipeline for testing calculated bounding boxes.

    Compares calculated bounding boxes to ground truth,
    via visualization and intersection over union.
    """

    self.visualize_bounding_boxes("images/gold/gold-visualized",
                                  self.bb_gold_list)
    start = time.process_time()
    bb_generator = BBGenerator(self.parsed_image_dataset)
    bb_list = bb_generator.generate_random()
    print("Time taken to generate bounding boxes: " +
          str(time.process_time() - start) + " seconds")
    self.visualize_bounding_boxes("images/random/random-visualized", bb_list)
    ious = []
    for (bb, bb_gold) in zip(bb_list, self.bb_gold_list):
      ious.append(self.calculate_iou(bb, bb_gold))
    print("Average IOU: " + str(np.mean(ious)))
