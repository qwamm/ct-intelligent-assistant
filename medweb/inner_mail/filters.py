from django_filters import rest_framework as filters
from inner_mail import models


class NotificationDynamicsFilter(filters.FilterSet):
    class Meta:
        model = models.NotificationDynamics
        fields = ["status"]
