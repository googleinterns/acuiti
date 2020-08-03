"""This module contains a CorrectnessMetrics class."""
import dataclasses


@dataclasses.dataclass
class CorrectnessMetrics:
  """Class for keeping track of the correctness results of an experiment.

   Correctness is measured in terms of accuracy, precision, and recall.
  """
  accuracy: float
  precision: float
  recall: float

  def __init__(self, accuracy: float, precision: float, recall: float):
    self.accuracy = accuracy
    self.precision = precision
    self.recall = recall
