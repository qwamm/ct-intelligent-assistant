from rest_framework import serializers as ser
from django.core.validators import EmailValidator

from medml.models import (
    MedWorker,
    Patient,
    PatientCard,
    UZIImage,
    UZIDevice,
    OriginalImage,
    SegmentationData,
    SegmentationPoint,
    UZISegmentGroupInfo,
    MLModel,
    dcm_validator,
)
from medml import utils
from medml.json_base.forms.UZIGroupForm import (
    UZIFormUpdate,
    UZIForm,
    UZINullForm,
    UZISegmentationDataForm,
    UZISegmentationGroupForm,
)

from django.db.models import F, Value, JSONField
from django.db.models.expressions import CombinedExpression
from django.forms.models import model_to_dict
from django.db import transaction
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError

"""MIXINS"""


class RelativeURLMixin:
    def to_representation(self, instance: OriginalImage):
        response = super(RelativeURLMixin, self).to_representation(instance)
        if instance.image:
            response["image"] = instance.image.tiff_file_url
            response["image_original"] = instance.image.url
            response["image_count"] = instance.image.pngs_len
            response["slide_template"] = instance.image.slide_template
        return response


"""Patients' Serializers"""


class PatientSerializer(ser.ModelSerializer):
    class Meta:
        model = Patient
        fields = "__all__"
        extra_kwargs = {"email": {"validators": [EmailValidator()]}}

    def create(self, validated_data):
        try:
            obj = Patient.objects.get(email=validated_data["email"])
        except:
            obj = Patient.objects.create(**validated_data)
        return obj


class UZIDeviceSerializer(ser.ModelSerializer):
    class Meta:
        model = UZIDevice
        fields = "__all__"


"""MedWorkers' Serializers"""


class MedWorkerRegistrationSerializer(ser.ModelSerializer):
    password1 = ser.CharField(
        write_only=True,
        required=True,
        style={"input_type": "password", "placeholder": "Пароль"},
        validators=[validate_password]
    )

    password2 = ser.CharField(
        write_only=True,
        required=True,
        style={"input_type": "password", "placeholder": "Повторите Пароль"},
    )

    class Meta:
        model = MedWorker
        fields = [
            "email",
            "last_name",
            "first_name",
            "fathers_name",
            "med_organization",
            "password1",
            "password2",
        ]
        # add extra validators MinLength TypeCheck
        extra_kwargs = {
            "last_name": {"required": True},
            "first_name": {"required": True},
            "fathers_name": {"required": True},
            "med_organization": {"required": True},
        }

    def validate(self, attrs):
        """Дополнительная валидация паролей"""
        password1 = attrs.get('password1')
        password2 = attrs.get('password2')

        if password1 != password2:
            raise ValidationError({"password2": "Пароли должны совпадать"})

        if len(password1) < 8:
            raise ValidationError({"password1": "Пароль должен содержать минимум 8 символов"})

        return attrs

    def create(self, validated_data: dict):
        password1 = validated_data.pop("password1")
        password2 = validated_data.pop("password2")
        user: MedWorker = MedWorker.objects.create_user(
            **validated_data, password=password1
        )
        return user


class MedWorkerCommonSerializer(ser.ModelSerializer):
    class Meta:
        model = MedWorker
        fields = [
            "last_name",
            "first_name",
            "fathers_name",
            "med_organization",
            "job",
            "is_remote_worker",
            "expert_details",
            "id",
        ]

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        if not ret["is_remote_worker"]:
            ret["expert_details"] = ""
        return ret


class PatientCardItemSerializer(ser.ModelSerializer):
    """UNUSED"""

    class Meta:
        model = PatientCard
        exclude = ["med_worker"]


class MedWorkerTableSerializer(ser.Serializer):
    """
    Read-only serializer
    UNUSED
    """

    cards = PatientCardItemSerializer(many=True)
    med_worker = MedWorkerCommonSerializer()

    def __new__(cls, *args, **kwargs):
        kwargs["many"] = False
        return super(MedWorkerTableSerializer, cls).__new__(
            cls, *args, **kwargs
        )

    def to_representation(self, instance):
        med_worker = self.context["medworker"]
        patients = {}
        for c in instance:
            if c.patient.id not in patients:
                patients[c.patient.id] = PatientSerializer().to_representation(
                    c.patient
                )
        ret = super().to_representation(
            {"cards": instance, "med_worker": med_worker}
        )
        ret["patients"] = patients
        return ret


class PatientCardTableItemSerializer(ser.ModelSerializer):
    patient = PatientSerializer()

    class Meta:
        model = PatientCard
        exclude = ["med_worker"]

    def to_representation(self, instance):
        return super().to_representation(instance)


class MedWorkerPatientsTableSerializer(ser.Serializer):
    cards = PatientCardTableItemSerializer(many=True)
    med_worker = MedWorkerCommonSerializer()

    def __new__(cls, *args, **kwargs):
        kwargs["many"] = False
        return super(MedWorkerPatientsTableSerializer, cls).__new__(
            cls, *args, **kwargs
        )

    def to_representation(self, instance):
        med_worker = self.context["medworker"]
        ret = super().to_representation(
            {"cards": instance, "med_worker": med_worker}
        )
        return ret


"""UIZs' serializers"""


class ShotsSerializer(ser.ModelSerializer):
    class Meta:
        model = PatientCard
        exclude = ["patient", "med_worker", "id"]


class AiInfoSerializer(ser.ModelSerializer):
    class Meta:
        model = UZISegmentGroupInfo
        fields = ["details", "is_ai", "id"]

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret.update(ret.pop("details", {}))
        return ret


class UZIImageModelSerializer(ser.ModelSerializer):
    patient_card = ShotsSerializer()

    class Meta:
        model = UZIImage
        fields = "__all__"

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret["uzi_device"] = getattr(instance.uzi_device, "name", None)
        ai_info = AiInfoSerializer(instance=instance.image.segments, many=True)
        ret["details"]["ai_info"] = ai_info.data
        return ret


"""Patients' serializers"""


class PatientCardSerializer(ser.ModelSerializer):
    class Meta:
        model = PatientCard
        exclude = ["patient", "med_worker"]


class PatientCardDefaultSerializer(ser.ModelSerializer):
    class Meta:
        model = PatientCard
        fields = "__all__"


class PatientAndCardSerializer(ser.Serializer):
    patient = PatientSerializer()
    card = PatientCardSerializer()

    def create(self, validated_data):
        med_worker = self.context["med_worker"]
        patient = self.fields["patient"].create(validated_data["patient"])
        card_data: dict = validated_data["card"]
        card_data["med_worker"] = med_worker
        card_data["patient"] = patient
        card = self.fields["card"].create(card_data)
        return {"patient": patient, "card": card}

    def update(self, instance, validated_data):
        patient = instance["patient"]
        card = instance["card"]
        self.inst_update(patient, validated_data["patient"])
        self.inst_update(card, validated_data["card"])
        patient.save()
        card.save()
        return {"patient": patient, "card": card}

    def inst_update(self, ins, values):
        for i in values:
            setattr(ins, i, values[i])


class PatientTableSerializer(ser.Serializer):
    patient = PatientSerializer()
    shots = UZIImageModelSerializer(many=True)

    def __new__(cls, *args, **kwargs):
        kwargs["many"] = False
        return super(PatientTableSerializer, cls).__new__(cls, *args, **kwargs)

    def to_representation(self, instance):
        patient = self.context["patient"]
        ret = super().to_representation(
            {"shots": instance, "patient": patient}
        )
        return ret


"""UZIS' serizlisers"""


class UZIImageCreateSerializer(ser.ModelSerializer):
    original_image = ser.FileField(
        required=True, write_only=True, validators=[dcm_validator]
    )
    projection_type = ser.ChoiceField(
        choices=UZIImage.PROJECTION_TYPE_CHOICES,
        default=UZIImage.PROJECTION_TYPE_CHOICES[0][0],
    )

    class Meta:
        model = UZIImage
        fields = (
            "uzi_device",
            "projection_type",
            "patient_card",
            "original_image",
        )

    def create(self, validated_data):
        image = validated_data.pop("original_image")
        # nimage, count = utils.in_mem_image_pre_saver(image)
        ssr = UZIForm(
            data={"projection_type": validated_data.pop("projection_type")}
        )  # TODO: CHANGE FORM
        ssr.is_valid(raise_exception=True)
        validated_data["details"] = ssr.validated_data
        # validated_data["image_count"] = count
        originaImage = OriginalImage.objects.create(image=image)
        validated_data["image"] = originaImage
        uzi_image = super().create(validated_data)
        return {"uzi_image": uzi_image, "image": originaImage}


class UZIImageCreate2Serializer(ser.ModelSerializer):
    original_image = ser.FileField(
        required=True, write_only=True, validators=[dcm_validator]
    )
    projection_type = ser.ChoiceField(
        choices=UZIImage.PROJECTION_TYPE_CHOICES,
        default=UZIImage.PROJECTION_TYPE_CHOICES[0][0],
    )
    # details = UZINullForm(required=False) # TODO: change if u want to add some details to UZIImage

    class Meta:
        model = UZIImage
        fields = (
            "uzi_device",
            "projection_type",
            "patient_card",
            "original_image",
        )

    def create(self, validated_data):
        # TODO: CHANGE to return ID of image_group
        image = validated_data.pop("original_image")
        nimage, count = utils.in_mem_image_pre_saver(image)
        ssr = UZINullForm(
            data={"projection_type": validated_data.pop("projection_type")}
        )  # TODO: CHANGE FORM
        ssr.is_valid(raise_exception=True)
        validated_data["details"] = ssr.validated_data
        image_group = super().create(validated_data)
        originaImage = OriginalImage.objects.create(
            image=nimage, image_group=image_group, image_count=count
        )
        return {"image_group": image_group, "image": originaImage}


class UZIOriginalImageSerializer(RelativeURLMixin, ser.ModelSerializer):

    def to_representation(self, instance):
        return super().to_representation(instance)

    class Meta:
        model = OriginalImage
        fields = "__all__"


class UZIImageAllSerializer(ser.ModelSerializer):
    class Meta:
        model = UZIImage
        exclude = ["uzi_device", "patient_card"]


class UZIImageSupprotSerializer(ser.Serializer):
    patient = PatientSerializer()
    uzi_device = UZIDeviceSerializer()
    patient_card = PatientCardSerializer()
    image_group = UZIImageAllSerializer()

    UZI_DEV_PREFIX = "uzi_device_"

    def to_representation(self, obj: UZIImage):
        sz = super().to_representation({
            "patient": obj.patient_card.patient,
            "uzi_device": obj.uzi_device,
            "patient_card": obj.patient_card,
            "image_group": obj,
        })
        patient_card = sz.pop("patient_card")
        uzi_device = sz.pop("uzi_device")
        image_group = sz.pop("image_group")
        sz.update(
            {f"{self.UZI_DEV_PREFIX}{f}": uzi_device[f] for f in uzi_device}
        )
        sz.update(patient_card)
        sz["patient_card_id"] = sz.pop("id")
        sz.update(image_group)
        return sz


class UZISegmentationPointSerializer(ser.ModelSerializer):
    class Meta:
        model = SegmentationPoint
        exclude = ["segment"]


class UZISegmentationDataPointsSerializer(ser.ModelSerializer):
    points = UZISegmentationPointSerializer(many=True)
    details = UZISegmentationDataForm()

    def to_representation(self, instance):
        points = instance.points.all()
        tmp = model_to_dict(instance)
        tmp["points"] = points
        ret = super().to_representation(tmp)
        return ret

    class Meta:
        model = SegmentationData
        # fields = '__all__'
        exclude = ["segment_group"]


class UZISegmentationDataSerializer(ser.ModelSerializer):
    data = UZISegmentationDataPointsSerializer(many=True)
    details = UZISegmentationGroupForm()

    def to_representation(self, instance):
        seg_data = instance.data.all()
        tmp = model_to_dict(instance)
        tmp["data"] = seg_data
        ret = super().to_representation(tmp)
        return ret

    class Meta:
        model = UZISegmentGroupInfo
        exclude = ["original_image"]


class UZIImageGetSerializer(ser.Serializer):
    image = UZIOriginalImageSerializer()
    segmentation = UZISegmentationDataSerializer(many=True)
    info = UZIImageSupprotSerializer()

    def to_representation(self, instance):
        image = getattr(instance, "image", None)
        segmentation = None
        if image is not None:
            segmentation = image.segments.all()
        data = {"image": image, "segmentation": segmentation, "info": instance}
        return super().to_representation(data)


class PatientCardUpdateSerializer(ser.ModelSerializer):
    class Meta:
        model = PatientCard
        fields = ["patient", "acceptance_datetime", "has_nodules", "diagnosis"]
        extra_kwargs = {
            "patient": {"required": False, "allow_null": False},
            "has_nodules": {"required": False},
            "diagnosis": {"required": False},
            "acceptance_datetime": {"required": False},
        }


class UZIShowUpdateSerializer(ser.ModelSerializer):
    patient_card = PatientCardUpdateSerializer()
    details = UZIFormUpdate(required=False)

    class Meta:
        model = UZIImage
        exclude = ["image_count"]
        extra_kwargs = {
            "uzi_device": {"required": False, "allow_null": False},
            "brightness": {"required": False, "allow_null": False},
            "contrast": {"required": False, "allow_null": False},
            "sharpness": {"required": False, "allow_null": False},
            "diagnos_date": {"required": False, "allow_null": False},
            "image": {"required": False, "allow_null": False},
        }

    def update(self, instance, validated_data):
        patient_card = validated_data.pop("patient_card", None)
        details = validated_data.get("details", {})

        validated_data["details"] = CombinedExpression(
            F("details"), "||", Value(details, JSONField())
        )
        if patient_card is not None:
            PatientCardUpdateSerializer().update(
                instance.patient_card, patient_card
            )
        ret = super().update(instance, validated_data)
        ret.details = {}
        return ret


class UZIImageInfoSerialier(ser.Serializer):
    info = UZIImageSupprotSerializer()


class UZIImageUpdateSerializer(ser.ModelSerializer):
    class Meta:
        model = UZIImage
        fields = ["pk"]


class UZIUpdateOriginalImageSerializer(ser.ModelSerializer):
    class Meta:
        model = UZIImage
        exclude = ["image", "image_count"]


class UZISegmentationGroupBaseSerializer(ser.ModelSerializer):
    class Meta:
        model = UZISegmentGroupInfo
        exclude = ["original_image"]
        extra_kwargs = {"is_ai": {"read_only": True}}

    def create(self, validated_data):
        # creating new segment group
        validated_data["details"]["is_ai"] = False
        data = validated_data.pop("data") or []
        validated_data["original_image_id"] = self.context["view"].kwargs[
            "img_id"
        ]
        segment_group = super().create(validated_data)
        return segment_group


class UZISegmentationPointCreateSerializer(ser.ModelSerializer):
    class Meta:
        model = SegmentationPoint
        exclude = ["segment"]
        extra_kwargs = {"uid": {"read_only": True}}


class UZISegmentationDataCreateSerializer(ser.ModelSerializer):
    points = UZISegmentationPointCreateSerializer(many=True)

    class Meta:
        model = SegmentationData
        fields = ["points"]

    def to_internal_value(self, data):
        print(f"{data=}")
        return super().to_internal_value(data)


class UZISegmentationGroupCreateSerializer(ser.ModelSerializer):
    details = UZISegmentationGroupForm()
    data = UZISegmentationDataCreateSerializer()

    class Meta:
        model = UZISegmentGroupInfo
        exclude = ["original_image", "is_ai"]

    def create(self, validated_data):
        # creating new segment group
        validated_data["details"]["is_ai"] = False
        data = validated_data.pop("data") or []
        validated_data["original_image_id"] = self.context["view"].kwargs[
            "uzi_img_id"
        ]
        segment_group = super().create(validated_data)
        # creating new segment
        points = data["points"]
        seg_object = SegmentationData.objects.create(
            segment_group=segment_group, details=validated_data["details"]
        )
        # creating new points
        inst_points = []
        for i, pi in enumerate(points):
            pi["segment"] = seg_object
            pi["uid"] = i
            inst_points.append(SegmentationPoint(**pi))
        SegmentationPoint.objects.bulk_create(inst_points)
        ret = model_to_dict(segment_group)
        ret["data"] = data
        return ret


class UZISegmentationGroupCreateSoloSerializer(ser.ModelSerializer):
    details = UZISegmentationGroupForm()

    class Meta:
        model = UZISegmentGroupInfo
        exclude = ["original_image", "is_ai"]

    def create(self, validated_data):
        validated_data["is_ai"] = False
        validated_data["original_image_id"] = self.context["view"].kwargs[
            "uzi_img_id"
        ]
        return super().create(validated_data)


class UZISegmentationGroupUpdateDeleteSerializer(ser.ModelSerializer):
    details = UZISegmentationGroupForm()

    class Meta:
        model = UZISegmentGroupInfo
        exclude = ["original_image", "is_ai"]

    def update(self, instance, validated_data):
        instance.details.update(validated_data["details"])
        instance.save()
        return instance


class UZISegmentationAddSerializer(ser.ModelSerializer):
    points = UZISegmentationPointCreateSerializer(many=True)

    class Meta:
        model = SegmentationData
        fields = ["points", "segment_group", "id"]
        extra_kwargs = {
            "points": {"write_only": True},
            "id": {"read_only": True},
        }

    def create(self, validated_data):
        # creating new segment
        print(f"{validated_data=}")
        points = validated_data.pop("points") or []
        seg_object = super().create(validated_data)
        # creating new points
        inst_points = []
        for i, pi in enumerate(points):
            pi["segment"] = seg_object
            pi["uid"] = i
            inst_points.append(SegmentationPoint(**pi))
        SegmentationPoint.objects.bulk_create(inst_points)
        return seg_object


class UZISegmentationUpdateDeleteSerializer(ser.ModelSerializer):
    points = UZISegmentationPointCreateSerializer(many=True)

    class Meta:
        model = SegmentationData
        fields = ["points", "segment_group"]
        extra_kwargs = {"segment_group": {"required": False}}

    def update(self, instance, validated_data):
        instance.segment_group = validated_data.get(
            "segment_group", instance.segment_group
        )

        points = validated_data.pop("points") or []
        with transaction.atomic():
            if points:
                SegmentationPoint.objects.filter(segment=instance).delete()
                inst_points = []
                for i, pi in enumerate(points):
                    pi["segment"] = instance
                    pi["uid"] = i
                    inst_points.append(SegmentationPoint(**pi))
                SegmentationPoint.objects.bulk_create(inst_points)

        return super().update(instance, validated_data)


"""ML"""


class MLModelSerializer(ser.ModelSerializer):
    class Meta:
        model = MLModel
        fields = "__all__"


"""Patient"""


class UZIImagePatientCreateSerializer(ser.Serializer):
    shot_data = UZIImageCreateSerializer()
    email = ser.EmailField()

    def create(self, validated_data):
        email = validated_data["email"]
        aaa = self.fields["shot_data"]
        ret = aaa.create(validated_data=validated_data["shot_data"])
        ret.update({"email": email})
        return ret
