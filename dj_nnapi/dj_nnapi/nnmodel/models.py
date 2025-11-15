import uuid

from django.db import models
from django.utils import timezone


class OriginalImage(models.Model):
    id = models.BigAutoField(primary_key=True)
    create_date = models.DateTimeField()
    delay_time = models.FloatField()
    viewed_flag = models.BooleanField()
    image = models.CharField(max_length=100)

    class Meta:
        verbose_name = "Снимок оригинала"
        verbose_name_plural = "Снимки оригиналов"
        managed = False
        db_table = "medml_originalimage"


class UZISegmentGroupInfo(models.Model):
    details = models.JSONField(null=True)

    is_ai = models.BooleanField(default=False)

    original_image = models.ForeignKey(
        OriginalImage, models.CASCADE, related_name="segments"
    )

    #   uzi_image = models.ForeignKey(
    #     'MedmlUziimage',
    #     on_delete=models.CASCADE
    #   )

    class Meta:
        managed = False
        verbose_name = "Информация о группе сегментов"
        verbose_name_plural = "Информация о группе сегментов"
        db_table = "nnmodel_uzisegmentgroupinfo"


class SegmentationData(models.Model):
    details = models.JSONField(null=True)

    segment_group = models.ForeignKey(
        UZISegmentGroupInfo, models.CASCADE, related_name="data"
    )

    class Meta:
        verbose_name = "Сегмент"
        verbose_name_plural = "Сегменты"
        managed = False
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
        managed = False
        db_table = "nnmodel_segmentationpoint"
        verbose_name = "Точка сегмента"
        verbose_name_plural = "Точки сегментов"
        unique_together = (["uid", "segment"],)


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = "auth_group"


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey("AuthPermission", models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = "auth_group_permissions"
        unique_together = (("group", "permission"),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey("DjangoContentType", models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = "auth_permission"
        unique_together = (("content_type", "codename"),)


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.SmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey(
        "DjangoContentType", models.DO_NOTHING, blank=True, null=True
    )
    user = models.ForeignKey("MedmlMedworker", models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = "django_admin_log"


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = "django_content_type"
        unique_together = (("app_label", "model"),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "django_migrations"


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "django_session"


class InnerMailMaildetails(models.Model):
    id = models.BigAutoField(primary_key=True)
    msg = models.TextField()
    mail_type = models.IntegerField()
    nodule_type = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = "inner_mail_maildetails"


class InnerMailNotification(models.Model):
    id = models.BigAutoField(primary_key=True)
    create_date = models.DateTimeField()
    details = models.ForeignKey(InnerMailMaildetails, models.DO_NOTHING)
    notification_author = models.ForeignKey(
        "MedmlMedworker", models.DO_NOTHING
    )
    notification_group = models.ForeignKey(
        "InnerMailNotificationgroup", models.DO_NOTHING
    )

    class Meta:
        managed = False
        db_table = "inner_mail_notification"


class InnerMailNotificationdynamics(models.Model):
    id = models.BigAutoField(primary_key=True)
    status = models.IntegerField()
    update_date = models.DateTimeField()
    mail = models.ForeignKey(InnerMailNotification, models.DO_NOTHING)
    user = models.ForeignKey("MedmlMedworker", models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = "inner_mail_notificationdynamics"
        unique_together = (("mail", "status", "user"),)


class InnerMailNotificationgroup(models.Model):
    id = models.BigAutoField(primary_key=True)
    title = models.CharField(max_length=512)
    create_date = models.DateTimeField()
    uzi_patient_card = models.ForeignKey(
        "MedmlPatientcard", models.DO_NOTHING, blank=True, null=True
    )

    class Meta:
        managed = False
        db_table = "inner_mail_notificationgroup"


class InnerMailNotificationgroupMembers(models.Model):
    id = models.BigAutoField(primary_key=True)
    notificationgroup = models.ForeignKey(
        InnerMailNotificationgroup, models.DO_NOTHING
    )
    medworker = models.ForeignKey("MedmlMedworker", models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = "inner_mail_notificationgroup_members"
        unique_together = (("notificationgroup", "medworker"),)


class MedmlMedworker(models.Model):
    id = models.BigAutoField(primary_key=True)
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()
    email = models.CharField(unique=True, max_length=254)
    is_remote_worker = models.BooleanField()
    fathers_name = models.CharField(max_length=150)
    med_organization = models.CharField(max_length=512)
    job = models.CharField(max_length=256, blank=True, null=True)
    expert_details = models.TextField()

    class Meta:
        managed = False
        db_table = "medml_medworker"


class MedmlMedworkerGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    medworker = models.ForeignKey(MedmlMedworker, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = "medml_medworker_groups"
        unique_together = (("medworker", "group"),)


class MedmlMedworkerUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    medworker = models.ForeignKey(MedmlMedworker, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = "medml_medworker_user_permissions"
        unique_together = (("medworker", "permission"),)


class MedmlMlmodel(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=256)
    file = models.CharField(max_length=100)
    model_type = models.CharField(max_length=1)
    projection_type = models.CharField(max_length=10)

    class Meta:
        managed = False
        db_table = "medml_mlmodel"


class MedmlPatient(models.Model):
    id = models.BigAutoField(primary_key=True)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    fathers_name = models.CharField(max_length=150)
    personal_policy = models.CharField(max_length=16)
    email = models.CharField(unique=True, max_length=254)
    is_active = models.BooleanField()

    class Meta:
        managed = False
        db_table = "medml_patient"


class MedmlPatientcard(models.Model):
    id = models.BigAutoField(primary_key=True)
    acceptance_datetime = models.DateTimeField()
    has_nodules = models.CharField(max_length=128)
    diagnosis = models.TextField()
    med_worker = models.ForeignKey(
        MedmlMedworker, models.DO_NOTHING, blank=True, null=True
    )
    patient = models.ForeignKey(
        MedmlPatient, models.DO_NOTHING, blank=True, null=True
    )

    class Meta:
        managed = False
        db_table = "medml_patientcard"


class MedmlUzidevice(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=512)

    class Meta:
        managed = False
        db_table = "medml_uzidevice"


class MedmlUziimage(models.Model):
    id = models.BigAutoField(primary_key=True)
    brightness = models.FloatField()
    contrast = models.FloatField()
    sharpness = models.FloatField()
    image_count = models.IntegerField()
    details = models.JSONField()
    image = models.OneToOneField(
        OriginalImage,
        models.DO_NOTHING,
        blank=True,
        null=True,
        related_name="uzi_image",
    )
    diagnos_date = models.DateTimeField(auto_now=True)

    patient_card = models.ForeignKey(
        MedmlPatientcard, models.DO_NOTHING, blank=True, null=True
    )
    uzi_device = models.ForeignKey(
        MedmlUzidevice, models.DO_NOTHING, blank=True, null=True
    )

    class Meta:
        managed = False
        db_table = "medml_uziimage"
