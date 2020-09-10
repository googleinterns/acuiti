"""This module contains a ConfusionMatrix class."""
import dataclasses
from modules import defaults
from modules.correctness_metrics import CorrectnessMetrics


@dataclasses.dataclass
class ConfusionMatrix:
  """Class for keeping track of the confusion matrix of an experiment.

   The confusion matrix contains number of false pos/neg, and true pos/neg
  """
  false_pos = int
  false_neg = int
  true_pos = int
  true_neg = int

  def __init__(self, false_pos: int, false_neg: int, true_pos: int,
               true_neg: int):
    self.false_pos = false_pos
    self.false_neg = false_neg
    self.true_pos = true_pos
    self.true_neg = true_neg

  def __add__(self, other):
    return ConfusionMatrix(self.false_pos + other.false_pos,
                           self.false_neg + other.false_neg,
                           self.true_pos + other.true_pos,
                           self.true_neg + other.true_neg)

  def calculate_correctness_metrics(self,
                                    output_path: str = defaults.OUTPUT_PATH
                                    ) -> CorrectnessMetrics:
    """Calculate the accuracy, precision, and recall for the confusion matrix.

    Arguments:
        output_path: if not None, prints accuracy, precision, and recall
        to file at path.

    Returns:
        CorrectnessMetrics dataclass representing (accuracy, precision, recall)
    """
    accuracy = (self.true_pos + self.true_neg) / (
        self.true_pos + self.true_neg + self.false_pos + self.false_neg)
    if self.true_pos == 0 and self.false_pos == 0:
      precision = 1
    else:
      precision = self.true_pos / (self.true_pos + self.false_pos)
    if self.true_pos == 0 and self.false_neg == 0:
      recall = 1
    else:
      recall = self.true_pos / (self.true_pos + self.false_neg)

    if output_path:
      with open(output_path, "a") as output_file:
        output_file.write("Accuracy: %f\n" % accuracy)
        output_file.write("Precision: %f\n" % precision)
        output_file.write("Recall: %f\n" % recall)

    print("Accuracy: %f\n" % accuracy)
    print("Precision: %f\n" % precision)
    print("Recall: %f\n\n" % recall)
    return CorrectnessMetrics(accuracy, precision, recall)
