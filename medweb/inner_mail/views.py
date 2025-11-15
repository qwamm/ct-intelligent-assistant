from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import PermissionDenied
from django.contrib.auth.models import AnonymousUser

from inner_mail import serializers as ser
from inner_mail import models
from inner_mail import filters

from django.db import connection, reset_queries

# len(connection.queries) - кол-во запросов к бд


class NotificationDynamicsGenerics(generics.ListAPIView):
    serializer_class = ser.NotificationDynamicsSerializer
    filterset_class = filters.NotificationDynamicsFilter

    def get_queryset(self):
        # user_id = self.request.user.id or 1
        user_id = self.request.user.id
        qs = (
            models.NotificationDynamics.objects.filter(user__id=user_id)
            .order_by("-update_date")
            .select_related(
                "mail", "mail__notification_group", "mail__details"
            )
        )
        return qs


class NotificationDynamicsViewedGenerics(generics.CreateAPIView):
    serializer_class = ser.NotificationDynamicsViewedSerializer

    def create(self, request, *args, **kwargs):
        # if request.user.id == request.data['user'] or 1:
        if request.user.id == request.data["user"]:
            a = super().create(request, *args, **kwargs)
            # print('len', len(connection.queries) - s)
            return a
        return Response(status=status.HTTP_403_FORBIDDEN)


class NotificationDynamicsCreateGenerics(generics.CreateAPIView):
    serializer_class = ser.NotificationDynamicsCreateSerializer

    def create(self, request, *args, **kwargs):
        # reset_queries()
        # s = len(connection.queries)
        a = super().create(request, *args, **kwargs)
        # print('len', len(connection.queries) - s)
        return a


class NotificationDynamicsReplyGenerics(generics.CreateAPIView):
    serializer_class = ser.NotificationDynamicsReplySerializer


class SimpleMailCreateGenerics(generics.CreateAPIView):
    serializer_class = ser.SimpleMailSerializer


class ExpertMailCreateGenerics(generics.CreateAPIView):
    serializer_class = ser.ExpertMailSerializer


class NotificationsGroupGenerics(generics.ListAPIView):
    serializer_class = ser.NotificationsOfGroupSerializer

    def get_queryset(self):
        # if isinstance(self.request.user, AnonymousUser) or not models.NotificationGroup.objects.filter(
        #   pk=self.kwargs.get('pk'),
        #   members__in=[self.request.user]
        # ).exists():
        #   raise PermissionDenied()

        qs = (
            models.Notification.objects.filter(
                notification_group=self.kwargs.get("pk")
            )
            .order_by("-create_date")
            .select_related("notification_author", "details")
        )
        return qs


class NotificationsGroupListGenerics(generics.ListAPIView):
    serializer_class = ser.NotificationGroupSerializer

    def get_queryset(self):
        qs = models.NotificationGroup.objects.filter(
            members__in=[self.request.user]
        ).order_by("-create_date")
        return qs
