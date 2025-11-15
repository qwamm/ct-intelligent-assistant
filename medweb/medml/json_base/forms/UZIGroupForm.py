from rest_framework.serializers import Serializer, ModelSerializer
from rest_framework import serializers as ser
from medml.models import UZIImage, SegmentationData, UZISegmentGroupInfo


class UZINullForm(Serializer):
    projection_type = ser.ChoiceField(
        choices=UZIImage.PROJECTION_TYPE_CHOICES,
        default=UZIImage.PROJECTION_TYPE_CHOICES[0][0],
    )


class UZIGroupForm(ModelSerializer):
    def __init__(self, instance=None, data=..., **kwargs):
        super().__init__(instance, data, **kwargs)

    projection_type = ser.ChoiceField(
        choices=UZIImage.PROJECTION_TYPE_CHOICES,
        default=UZIImage.PROJECTION_TYPE_CHOICES[0][0],
    )

    nodule_type = ser.IntegerField(
        min_value=1, max_value=5, default=1, allow_null=True
    )

    echo_descr = ser.CharField(default="", allow_null=True)

    nodule_1 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_2 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_3 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_4 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_5 = ser.FloatField(default=0, min_value=0, max_value=1)

    nodule_width = ser.FloatField(default=1, min_value=0)
    nodule_height = ser.FloatField(default=1, min_value=0)
    nodule_length = ser.FloatField(default=1, min_value=0)

    class Meta:
        model = UZIImage
        # fields = ['projection_type']
        exclude = ["details"]

    def create(self, validated_data):
        ll = set(
            [
                "projection_type",
                "nodule_type",
                "echo_descr",
                "nodule_1",
                "nodule_2",
                "nodule_3",
                "nodule_4",
                "nodule_5",
                "nodule_width",
                "nodule_height",
                "nodule_length",
            ]
        )
        details = {i: validated_data.pop("i") for i in ll}
        validated_data["details"] = details
        return super().create(validated_data)


class UZISegmentationAiForm(Serializer):
    nodule_type = ser.IntegerField(
        min_value=1, max_value=5, default=1, allow_null=True
    )

    nodule_2_3 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_4 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_5 = ser.FloatField(default=0, min_value=0, max_value=1)


class UZIFormUpdate(Serializer):
    projection_type = ser.ChoiceField(
        choices=UZIImage.PROJECTION_TYPE_CHOICES,
        default=UZIImage.PROJECTION_TYPE_CHOICES[0][0],
    )

    profile = ser.CharField(default="чёткие, ровные")

    right_length = ser.FloatField(min_value=0, required=False)
    right_width = ser.FloatField(min_value=0, required=False)
    right_depth = ser.FloatField(min_value=0, required=False)

    left_length = ser.FloatField(min_value=0, required=False)
    left_width = ser.FloatField(min_value=0, required=False)
    left_depth = ser.FloatField(min_value=0, required=False)

    isthmus = ser.FloatField(min_value=0, required=False)

    cdk = ser.CharField(required=False)
    position = ser.CharField(required=False)
    structure = ser.CharField(required=False)
    echogenicity = ser.CharField(required=False)

    additional_data = ser.CharField(required=False)
    rln = ser.CharField(required=False)
    result = ser.CharField(required=False)
    ai_info = UZISegmentationAiForm(required=False, many=True)


class UZIForm(Serializer):
    projection_type = ser.ChoiceField(
        choices=UZIImage.PROJECTION_TYPE_CHOICES,
        default=UZIImage.PROJECTION_TYPE_CHOICES[0][0],
    )

    profile = ser.CharField(default="чёткие, ровные")

    right_length = ser.FloatField(min_value=0, default=0)
    right_width = ser.FloatField(min_value=0, default=0)
    right_depth = ser.FloatField(min_value=0, default=0)

    left_length = ser.FloatField(min_value=0, default=0)
    left_width = ser.FloatField(min_value=0, default=0)
    left_depth = ser.FloatField(min_value=0, default=0)

    isthmus = ser.FloatField(min_value=0, default=0)

    cdk = ser.CharField(default="не измена")
    position = ser.CharField(default="обычное")
    structure = ser.CharField(default="однородная")
    echogenicity = ser.CharField(default="средняя")

    additional_data = ser.CharField(default="нет")
    rln = ser.CharField(default="нет")
    result = ser.CharField(default="без динамики")
    ai_info = UZISegmentationAiForm(required=False, many=True)


class UZISegmentationDataForm(ModelSerializer):
    # Специальная форма для информации о SegmentationData

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
        model = SegmentationData
        exclude = ["details", "segment_group"]


class UZISegmentationGroupForm(ModelSerializer):
    # Специальная форма для информации о SegmentationData

    nodule_type = ser.IntegerField(
        min_value=1, max_value=5, default=1, allow_null=True
    )

    nodule_2_3 = ser.FloatField(default=0.0, min_value=0, max_value=1)
    nodule_4 = ser.FloatField(default=0, min_value=0, max_value=1)
    nodule_5 = ser.FloatField(default=0.0, min_value=0, max_value=1)

    nodule_width = ser.FloatField(default=1.0, min_value=0)
    nodule_height = ser.FloatField(default=1.0, min_value=0)
    nodule_length = ser.FloatField(default=1.0, min_value=0)

    class Meta:
        model = UZISegmentGroupInfo
        exclude = ["details", "original_image"]
        extra_kwargs = {"is_ai": {"read_only": True}}


def segmetationDataForm(nn_class, isData=False):
    data = {
        "nodule_type": nn_class.argmax() + 3,
        "nodule_2_3": nn_class[0],
        "nodule_4": nn_class[1],
        "nodule_5": nn_class[2],
    }
    if isData:
        ser = UZISegmentationDataForm(data=data)
    else:
        ser = UZISegmentationGroupForm(data=data)
    ser.is_valid(raise_exception=True)
    return ser.validated_data
