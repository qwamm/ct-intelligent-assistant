from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from medml import models as med_models


class MailType(models.IntegerChoices):
    MSG = 0
    EXPERT_REPLY = 1


class MailDetails(models.Model):
    """
    Комментарии к сообщению к эксперту
    """

    msg = models.TextField("Комментарии", default="")

    mail_type = models.IntegerField(
        choices=MailType.choices, verbose_name="Тип соощения", default=0
    )

    nodule_type = models.IntegerField(
        "Тип узла",
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        default=1,
        null=True,
    )

    class Meta:
        verbose_name = "Комментарии к сообщению"
        verbose_name_plural = "Комментарии к сообщению"


class NotificationGroup(models.Model):
    title = models.CharField(
        "Заголовок уведомления", max_length=512, default=""
    )

    uzi_patient_card = models.ForeignKey(
        med_models.PatientCard,
        on_delete=models.CASCADE,
        verbose_name="Карта приема",
        null=True,
    )

    create_date = models.DateTimeField(
        verbose_name="Дата создания группы", auto_now_add=True
    )

    members = models.ManyToManyField(
        med_models.MedWorker,
        verbose_name="Участники",
        related_name="notif_members",
    )

    class Meta:
        verbose_name = "Внутренее уведомление"
        verbose_name_plural = "Внутрение уведомления"


class Notification(models.Model):
    notification_group = models.ForeignKey(
        NotificationGroup,
        models.CASCADE,
        verbose_name="Кому",
        related_name="notif_group",
    )

    notification_author = models.ForeignKey(
        med_models.MedWorker,
        models.CASCADE,
        verbose_name="От кого",
        related_name="notif_author",
    )

    details = models.ForeignKey(
        MailDetails,
        on_delete=models.CASCADE,
        verbose_name="Детали к сообщению",
        default="",
    )

    create_date = models.DateTimeField(
        verbose_name="Дата создания сообщения", auto_now_add=True
    )

    class Meta:
        verbose_name = "Уведомление"
        verbose_name_plural = "Уведомления"


class NotificationDynamics(models.Model):
    mail = models.ForeignKey(
        Notification,
        on_delete=models.CASCADE,
        verbose_name="Просмотренное Уведомление",
    )

    user = models.ForeignKey(
        med_models.MedWorker,
        on_delete=models.CASCADE,
        verbose_name="Кто просмотрел уведомление",
    )

    class MailStatus(models.IntegerChoices):
        NOT_VIEWED = 0
        VIEWED = 1

    status = models.IntegerField(
        choices=MailStatus.choices,
        verbose_name="Статус Уведомление",
        default=0,
    )

    update_date = models.DateTimeField("Дата изменения", auto_now=True)

    class Meta:
        verbose_name = "Динамика Уведомление"
        verbose_name_plural = "Динамика Уведомление"
        unique_together = ("mail", "status", "user")
