from django.apps import AppConfig
from typing import *
from .nn.nnmodel import ModelABC
from .nn.models.DetectionTrackingModel import DetectionTrackingModel
from .nn.models.ROISegmentationModel import ROISegmentationModel
from .nn.models.ROIClassificationModel import ROIClassificationModel
from os import getenv


class NNmodelConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "nnmodel"

    MODEL_DIR = {"S": "segUZI", "B": "boxUZI", "C": "classUZI"}

    wsgi = int(getenv("wsgi_start", "0"))

    DefalutModels: Dict[str, Dict[str, ModelABC]] = {
            "D": {"all": DetectionTrackingModel(model_type='all')},
            "C": {"all": ROIClassificationModel(model_type='all')},
            "S": {"all": ROISegmentationModel(model_type='all')}
    }
