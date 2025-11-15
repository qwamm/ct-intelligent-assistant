import json
import os
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile
from django.contrib import admin
from django.http import HttpResponse
from django_use_email_as_username.admin import BaseUserAdmin
from medml import models
import hashlib

from django.utils.translation import gettext_lazy as _

from medml.serializers import (
    UZISegmentationDataPointsSerializer,
    UZISegmentationGroupForm,
)
from rest_framework.serializers import DictField, CharField, ModelSerializer


class ZipUZISegmentationDataSerializer(ModelSerializer):
    data = UZISegmentationDataPointsSerializer(many=True)
    details = UZISegmentationGroupForm()
    original_path = CharField()
    uzi_details = DictField()
    uzi_device = CharField()

    class Meta:
        model = models.UZISegmentGroupInfo
        exclude = ["original_image"]


def download_slides(modeladmin, request, queryset, **extra):
    image_ids = queryset.values_list("id", flat=True)
    zip_hash = hashlib.sha256(
        ("_".join(map(str, image_ids))).encode("utf-8")
    ).hexdigest()[:16]
    groups = models.UZISegmentGroupInfo.objects.filter(
        original_image__id__in=image_ids, **extra
    ).prefetch_related(
        "data", "data__points", "original_image", "original_image__uzi_image"
    )
    for group in groups:
        setattr(
            group,
            "original_path",
            "model_files/" + os.path.basename(group.original_image.image.name),
        )
        uzi_image: models.UZIImage = group.original_image.uzi_image
        setattr(group, "uzi_details", uzi_image.details)
        setattr(group, "uzi_device", uzi_image.uzi_device.name)
    segs = ZipUZISegmentationDataSerializer(groups, many=True)

    with tempfile.SpooledTemporaryFile() as tmp:
        with ZipFile(tmp, "w", ZIP_DEFLATED) as uzi_zip:
            with uzi_zip.open("segments.json", "w") as json_file:
                json_data = json.dumps(segs.data, ensure_ascii=False)
                json_file.write(json_data.encode("utf-8"))
            for group in groups:
                with uzi_zip.open(group.original_path, "w") as img_out:
                    with group.original_image.image.file.open("rb") as img_inp:
                        img_out.write(img_inp.read())
        tmp.seek(0)
        response = HttpResponse(
            tmp.read(), content_type="application/x-zip-compressed"
        )
        response[
            "Content-Disposition"
        ] = f'attachment; filename="{zip_hash}.zip"'
        return response


@admin.action(description="Скачать снимки")
def download_all_slides(modeladmin, request, queryset):
    return download_slides(modeladmin, request, queryset)


@admin.action(description="Скачать отредактированные снимки")
def download_human_slides(modeladmin, request, queryset):
    return download_slides(modeladmin, request, queryset, is_ai=False)


class MedWorkerAdmin(BaseUserAdmin):
    """Define admin model for custom User model with no email field."""

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        (
            _("Personal info"),
            {"fields": ("first_name", "last_name", "fathers_name")},
        ),
        ("Медицинская организация", {"fields": ("med_organization", "job")}),
        (
            "Экспертная информация",
            {"fields": ("is_remote_worker", "expert_details")},
        ),
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                )
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "password1", "password2"),
            },
        ),
    )
    list_display = (
        "email",
        "first_name",
        "last_name",
        "fathers_name",
        "is_staff",
    )
    search_fields = ("email", "first_name", "last_name", "fathers_name")
    ordering = ("email",)


admin.site.register(models.MedWorker, MedWorkerAdmin)


@admin.register(models.Patient)
class PatientAdmin(admin.ModelAdmin):
    pass


@admin.register(models.UZIImage)
class UZIImageAdmin(admin.ModelAdmin):
    actions = [download_all_slides, download_human_slides]
    list_display = (
        "pk",
        "get_medworker",
        "get_patient",
        "uzi_device",
        "human_edited",
    )

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .prefetch_related(
                "uzi_device",
                "image__segments",
                "patient_card",
                "patient_card__patient",
                "patient_card__med_worker",
            )
        )

    def human_edited(self, obj):
        return obj.image.segments.filter(is_ai=False).exists()

    def get_medworker(self, obj):
        return obj.patient_card.med_worker.get_full_name()

    def get_patient(self, obj):
        return obj.patient_card.patient.get_full_name()

    human_edited.allow_tags = True
    human_edited.short_description = "Было отредактировано врачом?"
    get_medworker.short_description = "Лечащий врач"
    get_patient.short_description = "Пациент"


@admin.register(models.PatientCard)
class PatientCardAdmin(admin.ModelAdmin):
    pass


@admin.register(models.UZIDevice)
class UZIDeviceAdmin(admin.ModelAdmin):
    pass


@admin.register(models.OriginalImage)
class OriginalImageAdmin(admin.ModelAdmin):
    pass


@admin.register(models.SegmentationData)
class SegmentationDataAdmin(admin.ModelAdmin):
    pass


@admin.register(models.SegmentationPoint)
class SegmentationPointAdmin(admin.ModelAdmin):
    pass


@admin.register(models.MLModel)
class MLModelAdmin(admin.ModelAdmin):
    pass


@admin.register(models.UZISegmentGroupInfo)
class UZISegmentGroupInfoAdmin(admin.ModelAdmin):
    pass
