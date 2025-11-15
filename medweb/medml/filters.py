from django_filters import rest_framework as filters
from medml import models

from django.db.models import Value as V
from django.db.models.functions import Concat


def fullname_filter(qs, name, value):
    # 'fathers_name','first_name','last_name'
    vs = value.split()
    qs2 = qs.annotate(
        full_name=Concat(
            "last_name", V(" "), "first_name", V(" "), "fathers_name"
        )
    )
    for v in vs:
        qs2 = qs2.filter(full_name__icontains=v)
    return qs2


class MedWorkerListFilter(filters.FilterSet):
    # MedWorkerListView

    email__icontains = filters.CharFilter("email", "icontains")
    fullname = filters.CharFilter(method=fullname_filter, label="ФИО фильтр")

    class Meta:
        model = models.MedWorker
        fields = ["email", "is_remote_worker", "fullname"]


class PatientListFilter(filters.FilterSet):
    # PatientListView

    fullname = filters.CharFilter(method=fullname_filter, label="ФИО фильтр")
    email__icontains = filters.CharFilter("email", "icontains")
    personal_policy__icontains = filters.CharFilter(
        "personal_policy", "icontains"
    )

    class Meta:
        model = models.Patient
        fields = ["email__icontains", "personal_policy__icontains", "fullname"]


class SegmentGroupFilter(filters.FilterSet):
    class Meta:
        model = models.UZISegmentGroupInfo
        fields = ["is_ai"]
