# Icon Matching with Shape Context Descriptors

This project involves using shape-context descriptors to find an arbitrary template icon in a given image. The overall algorithm has three steps: 1) edge detection, 2) clustering contours, and 3) using shape context descriptors to find the closest contour to the template icon. Shape context descriptors were introduced in this 2001 research paper by S. Belongie et al: https://papers.nips.cc/paper/1913-shape-context-a-new-descriptor-for-shape-matching-and-object-recognition.pdf and its implementation is available in OpenCV: https://docs.opencv.org/master/d8/de3/classcv_1_1ShapeContextDistanceExtractor.html. Here's an overview of the process:

This project built and optimized the scale-and-color-invariant algorithm pipeline described above and achieved ~95% recall/precision in 1-2s on average.

**NOTE: This is not an officially supported Google product.**

# Repository Overview
Here's an overview of the files in this repository. There are three main types of files of interest (in bold) which are explained in further detail in each of the sections below.

Under ```modules/```:
  - **Benchmark Pipeline**, which runs any find icon algorithm on any dataset and outputs accuracy, latency, and memory information;
  - **Find Icon Algorithms**, which includes our optimized implementation of the shape context descriptor algorithm that achieves ~95% recall/precision in 1-2s on average;
  - **Analysis Utilities**, which are tools used to run experiments to figure out how to optimize our shape context descriptor algorithm;

Under ```tests/```:
- Integration and Unit Tests, which test the functionalities above.

Under ```datasets/```:
- Small datasets used for integration tests. Actual datasets to validate results are much larger and not included in this repository.
