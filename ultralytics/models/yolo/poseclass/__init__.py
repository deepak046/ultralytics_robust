"""PoseClass task components."""

from .predict import PoseClassPredictor
from .train import PoseClassTrainer
from .val import PoseClassValidator

__all__ = "PoseClassPredictor", "PoseClassTrainer", "PoseClassValidator"
