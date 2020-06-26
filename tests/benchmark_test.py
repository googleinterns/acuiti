import tensorflow as tf 
import numpy as np 
import cv2
from matplotlib import pyplot
import random
import time

image_feature_description = {
    'encoded_image_png': tf.io.FixedLenFeature([], tf.string),
    'encoded_icon_png': tf.io.FixedLenFeature([], tf.string),
    'box_ymin': tf.io.FixedLenFeature([], tf.float32),
    'box_xmin': tf.io.FixedLenFeature([], tf.float32),
    'box_ymax': tf.io.FixedLenFeature([], tf.float32),
    'box_xmax': tf.io.FixedLenFeature([], tf.float32),
}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

def parse_images(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    parsed_image_dataset = raw_dataset.map(_parse_image_function)
    return parsed_image_dataset

def visualize_bounding_boxes(parsed_image_dataset, output_name, bb_list = None):
    for i, image_features in enumerate(parsed_image_dataset):
        image_raw = image_features["encoded_image_png"].numpy()
        image_bgr = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), -1)
        if bb_list is None:
            ymin = image_features["box_ymin"]
            xmin = image_features["box_xmin"]
            ymax = image_features["box_ymax"]
            xmax = image_features["box_xmax"]
        else:
            bb = bb_list[i]
            ymin = bb['y1']
            xmin = bb['x1']
            ymax = bb['y2']
            xmax = bb['x2']
        # top left and bottom right corner of rectangle
        cv2.rectangle(image_bgr,(xmin,ymin),(xmax,ymax),(0,255,0),3)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        if image_rgb is None:
            print("Could not read the image.")
        pyplot.imshow(image_rgb)
        pyplot.imsave(output_name + str(i) + ".jpg", image_rgb)

def parse_bb_gold(parsed_image_dataset):
    bb_gold_list = []
    for image_features in parsed_image_dataset:
        bb_gold = {}
        bb_gold['y1'] = image_features["box_ymin"]
        bb_gold['x1'] = image_features["box_xmin"]
        bb_gold['y2'] = image_features["box_ymax"]
        bb_gold['x2'] = image_features["box_xmax"]
        bb_gold_list.append(bb_gold)
    return bb_gold_list
    

# This currently generates a random bounding box for the image, and 
# is simply a placeholder to ensure that the entire testing pipeline works.
def generate_bb(parsed_image_dataset):
    bb_list = []
    for i, image_features in enumerate(parsed_image_dataset):
        bb = {}
        image_raw = image_features["encoded_image_png"].numpy()
        image_bgr = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), -1)
        height = image_bgr.shape[0]
        width = image_bgr.shape[1]
        bb['y1'] = random.randint(0, height - 1)
        bb['x1'] = random.randint(0, width - 1)
        bb['x2'] = random.randint(bb['x1'], width - 1)
        bb['y2'] = random.randint(bb['y1'], height - 1)
        bb_list.append(bb)
    return bb_list 


# ref: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def calculate_bb_overlap(bb, bb_gold):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb['x1'], bb_gold['x1'])
    y_top = max(bb['y1'], bb_gold['y1'])
    x_right = min(bb['x2'], bb_gold['x2'])
    y_bottom = min(bb['y2'], bb_gold['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb_area = (bb['x2'] - bb['x1'] + 1) * (bb['y2'] - bb['y1'] + 1)
    bb_gold_area = (bb_gold['x2'] - bb_gold['x1'] + 1) * (bb_gold['y2'] - bb_gold['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb_area + bb_gold_area - intersection_area)
    return iou

# Integrated testing pipeline
relative_path = "acuiti/benchmark.tfrecord"
parsed_image_dataset = parse_images(relative_path)
visualize_bounding_boxes(parsed_image_dataset, "images/gold/gold-visualized")
bb_gold_list = parse_bb_gold(parsed_image_dataset)
start = time.process_time() 
bb_list = generate_bb(parsed_image_dataset)
print("Time taken to generate bounding boxes: " + str(time.process_time() - start) + " seconds")
visualize_bounding_boxes(parsed_image_dataset, "images/random/random-visualized", bb_list)
IOUs = []
for (bb, bb_gold) in zip(bb_list, bb_gold_list):
    IOUs.append(calculate_bb_overlap(bb, bb_gold))
print("Average IOU: " + str(np.mean(IOUs)))

            