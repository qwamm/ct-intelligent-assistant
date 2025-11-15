import uuid

from django.db import models
from django.utils.translation import gettext_lazy as _
from django_use_email_as_username.models import BaseUser, BaseUserManager
from django.core.validators import (
    RegexValidator,
    MinValueValidator,
    MaxValueValidator,
    FileExtensionValidator,
)
from django.utils.regex_helper import _lazy_re_compile
from django.utils import timezone

from medml import utils
from medml.fields import DicomAndTiffFileField


"""Mixins"""


def get_full_name(self):
    """
    Return the first_name plus the last_name, with a space in between.
    """
    full_name = "%s %s %s" % (
        self.last_name,
        self.first_name,
        self.fathers_name,
    )
    return full_name.strip().capitalize()


"""CUSTOM FIELDS"""


class PolicyField(models.CharField):
    def __init__(self, *args, **kwargs):
        kwargs["max_length"] = 16
        super(PolicyField, self).__init__(*args, **kwargs)
        self.validators.append(
            RegexValidator(
                _lazy_re_compile(r"^\d{16}\Z"),
                message=_("Enter a valid integer."),
                code="invalid",
            )
        )


class LowerEmailField(models.EmailField):
    def get_prep_value(self, value):
        return super().get_prep_value(value).lower()


dcm_validator = FileExtensionValidator(
    ["dcm", "png", "jpeg", "jpg", "tiff", "tif"]
)


"""DB MODELS"""


class MedWorker(BaseUser):
    """
    TODO: add patien create permissions
    """

    email = LowerEmailField(_("email address"), unique=True)

    is_remote_worker = models.BooleanField("Удаленный Эксперт", default=False)

    fathers_name = models.CharField("Отчество", max_length=150, blank=True)

    med_organization = models.CharField(
        "Медицинская организация", max_length=512, blank=True
    )

    job = models.CharField("Должность", max_length=256, blank=True, null=True)

    objects = BaseUserManager()

    expert_details = models.TextField(
        "Описание экспертных качеств", default=""
    )

    class Meta:
        permissions = [
            ("add_patient", "Can add Patient"),
            ("change_patient", "Can change Patient"),
            ("view_patient", "Can view Patient"),
        ]
        verbose_name = "Мед работник"
        verbose_name_plural = "Мед работники"

    def get_full_name(self) -> str:
        return get_full_name(self)


class UZIDevice(models.Model):
    """
    Аппарат, на котором происходило УЗИ диагностика
    """

    name = models.CharField("Название аппарата", max_length=512)

    class Meta:
        verbose_name = "Аппарат УЗИ"
        verbose_name_plural = "Аппараты УЗИ"

    def __str__(self) -> str:
        return f"{self.name}"


class PatientCard(models.Model):
    """
    Информация о том какой пациент посещал какого врача
    и результатах диагностики
    """

    NODULES_CHOICES = (
        ("T", "Обнаружено новообразование"),
        ("F", "Без паталогий"),
    )

    # Many2Many class
    patient = models.ForeignKey(
        "Patient", on_delete=models.SET_NULL, related_name="card", null=True
    )

    med_worker = models.ForeignKey(
        "MedWorker", on_delete=models.SET_NULL, related_name="card", null=True
    )

    acceptance_datetime = models.DateTimeField(
        "Дата и время приема", auto_now_add=True
    )

    has_nodules = models.CharField(
        "Узловые новообразования",
        max_length=128,
        choices=NODULES_CHOICES,
        default=NODULES_CHOICES[1][0],
    )

    diagnosis = models.TextField(_("Диагноз"), blank=True, default="")

    class Meta:
        verbose_name = "Карта пациента"
        verbose_name_plural = "Карты пациентов"


class OriginalImage(models.Model):
    create_date = models.DateTimeField("Дата создания", default=timezone.now)

    delay_time = models.FloatField("Время обработки", default=-1)

    viewed_flag = models.BooleanField("Просмотренно", default=False)

    image = DicomAndTiffFileField(
        "Cнимок", upload_to=utils.originalUZIPath, validators=[dcm_validator]
    )

    class Meta:
        verbose_name = "Снимок оригинала"
        verbose_name_plural = "Снимки оригиналов"


class SegmentationData(models.Model):
    details = models.JSONField(null=True)

    segment_group = models.ForeignKey(
        "UZISegmentGroupInfo", models.CASCADE, related_name="data"
    )

    class Meta:
        managed = True
        verbose_name = "Сегмент"
        verbose_name_plural = "Сегменты"
        db_table = "nnmodel_segmentationdata"


class SegmentationPoint(models.Model):
    #uid = models.UUIDField(primary_key = True, default = uuid.uuid4, editable = False)
    uid = models.BigIntegerField()

    segment = models.ForeignKey(
        SegmentationData, on_delete=models.CASCADE, related_name="points"
    )

    x = models.PositiveIntegerField()
    y = models.PositiveIntegerField()
    z = models.PositiveIntegerField(default=0)

    class Meta:
        managed = True
        verbose_name = "Точка сегмента"
        verbose_name_plural = "Точки сегментов"
        unique_together = (["uid", "segment"],)
        db_table = "nnmodel_segmentationpoint"


class UZISegmentGroupInfo(models.Model):
    details = models.JSONField(null=True)

    is_ai = models.BooleanField(default=False)

    original_image = models.ForeignKey(
        OriginalImage, models.CASCADE, related_name="segments"
    )

    class Meta:
        managed = True
        verbose_name = "Информация о группе сегментов"
        verbose_name_plural = "Информация о группе сегментов"
        db_table = "nnmodel_uzisegmentgroupinfo"


class UZIImage(models.Model):
    """
    УЗИ картинка пациента
    """

    PROJECTION_TYPE_CHOICES = (("long", "Продольный"), ("cross", "Поперечный"))

    brightness = models.FloatField(
        "Яркость",
        validators=[MinValueValidator(-1), MaxValueValidator(1)],
        default=0,
    )

    contrast = models.FloatField(
        "Контраст",
        validators=[MinValueValidator(-1), MaxValueValidator(1)],
        default=0,
    )

    sharpness = models.FloatField(
        "Резкость",
        validators=[MinValueValidator(-1), MaxValueValidator(1)],
        default=0,
    )

    image_count = models.IntegerField(
        "Количество снимков", validators=[MinValueValidator(0)], default=0
    )

    uzi_device = models.ForeignKey(
        "UZIDevice", on_delete=models.SET_NULL, null=True
    )

    patient_card = models.ForeignKey(
        "PatientCard",
        on_delete=models.SET_NULL,
        null=True,
        related_name="uzi_images",
    )

    diagnos_date = models.DateTimeField(auto_now=True)

    details = models.JSONField("Детали диагностики")

    image = models.OneToOneField(
        OriginalImage,
        verbose_name="Снимок",
        related_name="uzi_image",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )

    class Meta:
        verbose_name = "УЗИ зображение"
        verbose_name_plural = "УЗИ изображения"


class Patient(models.Model):
    first_name = models.CharField(_("first name"), max_length=150)
    last_name = models.CharField(_("last name"), max_length=150)
    fathers_name = models.CharField("Отчество", max_length=150)

    personal_policy = PolicyField("Полис")
    email = models.EmailField(_("email address"), unique=True)
    is_active = models.BooleanField(_("Пациент активен"), default=True)

    class Meta:
        verbose_name = "Пациент"
        verbose_name_plural = "Пациенты"

    def get_full_name(self) -> str:
        return get_full_name(self)

    def __str__(self):
        return self.get_full_name()


class MLModel(models.Model):
    MODEL_TYPES_CHOICES = (
        ("C", "Модель для классификации"),
        ("S", "Модель для сегментации"),
        ("B", "Модель для боксов"),
    )
    PROJECTION_TYPES_CHOICES = (
        ("cross", "поперечная"),
        ("long", "продольная"),
        ("all", "обе"),
    )

    name = models.CharField("Имя модели", max_length=256)
    file = models.FileField("Путь к файлу модели", upload_to=utils.mlModelPath)
    model_type = models.CharField(
        "Тип модели",
        choices=MODEL_TYPES_CHOICES,
        default=MODEL_TYPES_CHOICES[0][0],
        max_length=1,
    )
    projection_type = models.CharField(
        "Тип проекции",
        choices=PROJECTION_TYPES_CHOICES,
        default=PROJECTION_TYPES_CHOICES[0][0],
        max_length=10,
    )

    class Meta:
        verbose_name = "Модель МЛ"
        verbose_name_plural = "Модели МЛ"
