from rest_framework.serializers import Serializer, ModelSerializer
from rest_framework import serializers as ser
from medml.models import SegmentationData, CTSegmentGroupInfo

from medweb.medml.models import CTImage, CTSegmentGroupInfo


class CTNullForm(Serializer):
    projection_type = ser.ChoiceField(
        choices=CTImage.CT_TYPE_CHOICES,
        default=CTImage.CT_TYPE_CHOICES[0][0],
    )


class CTGroupForm(ModelSerializer):
    def __init__(self, instance=None, data=..., **kwargs):
        super().__init__(instance, data, **kwargs)

    ct_type = ser.ChoiceField(
        choices=CTImage.CT_TYPE_CHOICES,
        default=CTImage.CT_TYPE_CHOICES[0][0],
    )

    neoplasm_type = ser.IntegerField(
        min_value=1, max_value=3, default=1, allow_null=True
    )

    undefined_prob = ser.FloatField(default=0, min_value=0, max_value=1)
    malignant_prob = ser.FloatField(default=0, min_value=0, max_value=1)
    benign_prob = ser.FloatField(default=0, min_value=0, max_value=1)

    neoplasm_length = ser.FloatField(default=1, min_value=0)
    neoplasm_width = ser.FloatField(default=1, min_value=0)
    neoplasm_height = ser.FloatField(default=1, min_value=0)

    class Meta:
        model = CTImage
        # fields = ['projection_type']
        exclude = ["details"]

    def create(self, validated_data):
        ll = set(
            [
                "ct_type",
                "neoplasm_type",
                "undefined_prob",
                "malignant_prob",
                "benign_prob",
                "neoplasm_length",
                "neoplasm_height",
                "neoplasm_length",
            ]
        )
        details = {i: validated_data.pop("i") for i in ll}
        validated_data["details"] = details
        return super().create(validated_data)


class CTSegmentationAiForm(Serializer):
    neoplasm_type = ser.IntegerField(
        min_value=1, max_value=3, default=1, allow_null=True
    )

    undefined_prob = ser.FloatField(default=0, min_value=0, max_value=1)
    malignant_prob = ser.FloatField(default=0, min_value=0, max_value=1)
    benign_prob = ser.FloatField(default=0, min_value=0, max_value=1)


class CTFormUpdate(Serializer):
    projection_type = ser.ChoiceField(
        choices=CTImage.CT_TYPE_CHOICES,
        default=CTImage.CT_TYPE_CHOICES[0][0],
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
    ai_info = CTSegmentationAiForm(required=False, many=True)


class CTForm(Serializer):
    projection_type = ser.ChoiceField(
        choices=CTImage.CT_TYPE_CHOICES,
        default=CTImage.CT_TYPE_CHOICES[0][0],
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
    ai_info = CTSegmentationAiForm(required=False, many=True)


class CTSegmentationDataForm(ModelSerializer):
    # Специальная форма для информации о SegmentationData

    ct_type = ser.ChoiceField(
        choices=CTImage.CT_TYPE_CHOICES,
        default=CTImage.CT_TYPE_CHOICES[0][0],
    )

    neoplasm_type = ser.IntegerField(
        min_value=1, max_value=3, default=1, allow_null=True
    )

    undefined_prob = ser.FloatField(default=0, min_value=0, max_value=1)
    malignant_prob = ser.FloatField(default=0, min_value=0, max_value=1)
    benign_prob = ser.FloatField(default=0, min_value=0, max_value=1)

    neoplasm_length = ser.FloatField(default=1, min_value=0)
    neoplasm_width = ser.FloatField(default=1, min_value=0)
    neoplasm_height = ser.FloatField(default=1, min_value=0)

    class Meta:
        model = SegmentationData
        exclude = ["details", "segment_group"]


class CTSegmentationGroupForm(ModelSerializer):
    # Специальная форма для информации о SegmentationData

    ct_type = ser.ChoiceField(
        choices=CTImage.CT_TYPE_CHOICES,
        default=CTImage.CT_TYPE_CHOICES[0][0],
    )

    neoplasm_type = ser.IntegerField(
        min_value=1, max_value=3, default=1, allow_null=True
    )

    undefined_prob = ser.FloatField(default=0, min_value=0, max_value=1)
    benign_prob = ser.FloatField(default=0, min_value=0, max_value=1)
    malignant_prob = ser.FloatField(default=0, min_value=0, max_value=1)

    neoplasm_length = ser.FloatField(default=1, min_value=0)
    neoplasm_width = ser.FloatField(default=1, min_value=0)
    neoplasm_height = ser.FloatField(default=1, min_value=0)
    class Meta:
        model = CTSegmentGroupInfo
        exclude = ["details", "original_image"]
        extra_kwargs = {"is_ai": {"read_only": True}}


def segmetationDataForm(nn_class, isData=False):
    data = {
        "neoplasm_type": nn_class.argmax() + 3,
        "undefined_prob": nn_class[0],
        "benign_prob": nn_class[1],
        "malignant_prob": nn_class[2],
    }
    if isData:
        ser = CTSegmentationDataForm(data=data)
    else:
        ser = CTSegmentationGroupForm(data=data)
    ser.is_valid(raise_exception=True)
    return ser.validated_data
