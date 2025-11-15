from rest_framework import serializers as ser
from inner_mail import models
from inner_mail import utils
from medml import models as med_models
from medml import serializers as med_ser
from django.db.models import Q


class MailDetailsSerializer(ser.ModelSerializer):
    class Meta:
        model = models.MailDetails
        fields = "__all__"


class SimpleMailSerializer(ser.ModelSerializer):
    class Meta:
        model = models.MailDetails
        exclude = ["mail_type", "nodule_type"]

    def create(self, validated_data):
        validated_data["mail_type"] = models.MailType.MSG
        return super().create(validated_data)


class ExpertMailSerializer(ser.ModelSerializer):
    class Meta:
        model = models.MailDetails
        exclude = ["mail_type"]

    def create(self, validated_data):
        validated_data["mail_type"] = models.MailType.EXPERT_REPLY
        return super().create(validated_data)


class NotificationGroupSerializer(ser.ModelSerializer):
    class Meta:
        model = models.NotificationGroup
        fields = "__all__"


class NotificationGroupCreateSerializer(ser.ModelSerializer):
    members = ser.ListField(
        write_only=True, child=ser.IntegerField(min_value=0)
    )

    class Meta:
        model = models.NotificationGroup
        fields = "__all__"

    def create(self, validated_data):
        members = validated_data.pop("members")
        print(members)
        recipe = med_models.MedWorker.objects.filter(members__in=members)
        return recipe


class NotificationSerializer(ser.ModelSerializer):
    details = MailDetailsSerializer()
    notification_group = NotificationGroupSerializer()

    class Meta:
        model = models.Notification
        fields = "__all__"
        extra_kwargs = {"notification_author": {"read_only": True}}


class NotificationCreateSerializer(ser.ModelSerializer):
    details = MailDetailsSerializer()
    notification_group = NotificationGroupCreateSerializer()

    class Meta:
        model = models.Notification
        fields = "__all__"
        extra_kwargs = {"notification_author": {"read_only": True}}


class NotificationDynamicsSerializer(ser.ModelSerializer):
    mail = NotificationSerializer()

    class Meta:
        model = models.NotificationDynamics
        fields = "__all__"
        extra_kwargs = {
            "user": {"read_only": True},
            "status": {"read_only": True},
        }


from django.db import connection


class NotificationDynamicsCreateSerializer(ser.ModelSerializer):
    mail = NotificationCreateSerializer()

    class Meta:
        model = models.NotificationDynamics
        fields = "__all__"
        extra_kwargs = {
            "user": {"read_only": True},
            "status": {"read_only": True},
        }

    def create(self, validated_data):
        author = self.context["request"].user
        cc = set(validated_data["mail"]["notification_group"].pop("members"))
        # cc = set(int(idd) for idd in validated_data['mail']['notification_group'].pop('members'))
        membis = cc - set([author.pk])
        memb = med_models.MedWorker.objects.filter(id__in=membis)
        details = models.MailDetails.objects.create(
            **validated_data["mail"]["details"]
        )

        notification_group = models.NotificationGroup.objects.create(
            **validated_data["mail"]["notification_group"]
        )
        notification_group.members.set(memb)
        notification_group.members.add(author)
        mail = models.Notification.objects.create(
            details=details,
            notification_group=notification_group,
            notification_author=author,
        )

        validated_data["user"] = author
        validated_data[
            "status"
        ] = models.NotificationDynamics.MailStatus.VIEWED
        validated_data.pop("mail")
        mem_bulk = []
        for oid in memb:
            mem_bulk.append(
                models.NotificationDynamics(
                    mail=mail,
                    status=models.NotificationDynamics.MailStatus.NOT_VIEWED,
                    user=oid,
                )
            )
        notification_dynamics = models.NotificationDynamics(
            mail=mail, **validated_data
        )
        mem_bulk.append(notification_dynamics)
        models.NotificationDynamics.objects.bulk_create(mem_bulk)
        return notification_dynamics


class NotificationDynamicsViewedSerializer(ser.ModelSerializer):
    mail = ser.ListField(write_only=True, child=ser.IntegerField(min_value=0))

    class Meta:
        model = models.NotificationDynamics
        exclude = ["status"]
        # fields = ['mail','user']

    def create(self, validated_data):
        usr = validated_data["user"]
        cr1 = Q(
            mail__in=validated_data["mail"],
            user=usr,
            status=models.NotificationDynamics.MailStatus.NOT_VIEWED,
        )

        obj = models.NotificationDynamics.objects.filter(cr1).select_related(
            "mail"
        )
        objs = []
        for o in obj:
            o.status = models.NotificationDynamics.MailStatus.VIEWED
            objs.append(o)
        # print('objs len', len(objs))
        models.NotificationDynamics.objects.bulk_update(
            objs, fields=("status",)
        )
        return {"mail": objs, "user": usr}

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret["mails"] = [i.pk for i in instance["mail"]]
        return ret


class NotificationDynamicsReplySerializer(ser.ModelSerializer):
    class Meta:
        model = models.Notification
        fields = "__all__"
        # fields = ['notification_group', 'notification_author', 'details']

    def create(self, validated_data):
        notification_group = validated_data["notification_group"]
        # author = med_models.MedWorker.objects.get(id=1)
        author = self.context["request"].user
        details = validated_data["details"]

        notification = models.Notification.objects.create(
            notification_group=notification_group,
            notification_author=author,
            details=details,
        )
        members = notification_group.members.all()

        mem_bulk = []
        for oid in members:
            if oid != author:
                mem_bulk.append(
                    models.NotificationDynamics(
                        mail=notification,
                        user=oid,
                        status=models.NotificationDynamics.MailStatus.NOT_VIEWED,
                    )
                )
        mem_bulk.append(
            models.NotificationDynamics(
                mail=notification,
                user=author,
                status=models.NotificationDynamics.MailStatus.VIEWED,
            )
        )

        models.NotificationDynamics.objects.bulk_create(mem_bulk)

        return notification


class NotificationsOfGroupSerializer(ser.ModelSerializer):
    notification_author = med_ser.MedWorkerCommonSerializer()
    details = MailDetailsSerializer()

    class Meta:
        model = models.Notification
        fields = "__all__"


class NotificationsGroupListSerializser(ser.ModelSerializer):
    class Meta:
        model = models.NotificationGroup
        fields = "__all__"
