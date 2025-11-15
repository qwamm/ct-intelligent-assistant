import json
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework import status
from rest_framework.views import APIView
from . import models, tasks


# from nnmodel.nn.defaultModels import DefalutModels
from .apps import NNmodelConfig
from .nn.loaders.img_loader import defaultImgLoader
import cv2 as cv

"""SERIALIZERS"""
from rest_framework.serializers import Serializer, ModelSerializer
import rest_framework.serializers as ser


class PredictAllSerializer(Serializer):
    file_path = ser.CharField()
    projection_type = ser.CharField()
    id = ser.IntegerField()


class UZISegmentationForm(ModelSerializer):
    def __init__(self, instance=None, data=..., **kwargs):
        super().__init__(instance, data, **kwargs)

    nodule_type = ser.IntegerField(
        min_value=1, max_value=5, default=1, allow_null=True
    )

    nodule_2_3 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_4 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_5 = ser.FloatField(default=0, min_value=0, max_value=1)

    nodule_width = ser.FloatField(default=1, min_value=0)
    nodule_height = ser.FloatField(default=1, min_value=0)
    nodule_length = ser.FloatField(default=1, min_value=0)

    class Meta:
        model = models.SegmentationData
        exclude = ["details", "original_image"]

    def update(self, instance, validated_data):
        return super().update(instance, validated_data)


def segmetationDataForm(nn_class):
    data = {
        "nodule_type": nn_class.argmax() + 3,
        "nodule_2_3": nn_class[0],
        "nodule_4": nn_class[1],
        "nodule_5": nn_class[2],
    }
    ser = UZISegmentationForm(data=data)
    ser.is_valid(raise_exception=True)
    return ser.validated_data


"""APIVIEWS"""


class PredictAll(APIView):
    serializer_class = PredictAllSerializer

    def post(self, request: Request, *args, **kwargs):
        ser = self.serializer_class(data=request.data)
        ser.is_valid(raise_exception=True)
        k = tasks.predict_all(
            ser.validated_data["file_path"],
            ser.validated_data["projection_type"],
            ser.validated_data["id"],
        )
        return Response(
            data={"text": "predicted", "k": k}, status=status.HTTP_201_CREATED
        )
