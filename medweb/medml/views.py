import json

import concurrent.futures
from dramatiq.results.backends.redis import RedisBackend
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.permissions import AllowAny
from rest_framework import status
from rest_framework.generics import (
    CreateAPIView,
    UpdateAPIView,
    ListAPIView,
    RetrieveAPIView,
    RetrieveUpdateDestroyAPIView,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework import mixins
from rest_framework.viewsets import ModelViewSet

from django.db.models import Max, Prefetch
from django.http import Http404

from medweb.medml import filters
from medweb.medml import serializers as ser
from medweb.medml import models
from medweb.medml import tasks

"""MedWorkers' VIEWS"""


class RegistrationView(CreateAPIView):
    serializer_class = ser.MedWorkerRegistrationSerializer
    permission_classes = [AllowAny]


class MedWorkerChangeView(mixins.RetrieveModelMixin, UpdateAPIView):
    """
    Изменить информацию о мед работнике
    """

    serializer_class = ser.MedWorkerCommonSerializer

    def get_object(self):
        try:
            return models.MedWorker.objects.get(id=self.kwargs["id"])
        except:
            raise Http404

    def perform_update(self, serializer):
        return super().perform_update(serializer)

    def get(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.patch(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        if request.user.id is self.kwargs["id"]:
            return super().patch(request, *args, **kwargs)
        return Response(status=status.HTTP_403_FORBIDDEN)


class MedWorkerPatientsTableView(ListAPIView):
    """
    Для Start Page - инфа только о последенй карточке
    """

    serializer_class = ser.MedWorkerPatientsTableSerializer

    # TODO: add to mixin
    def get_medworker(self):
        try:
            self.medworker = models.MedWorker.objects.get(id=self.kwargs["id"])
            return self.medworker
        except:
            raise Http404

    def get_serializer_context(self):
        # TODO: remove one bd request
        medworker = self.get_medworker()
        ret = super().get_serializer_context()
        ret.update({"medworker": medworker})
        return ret

    def get_queryset(self):
        qs = models.PatientCard.objects.filter(
            med_worker__id=self.kwargs["id"]
        ).select_related("patient")
        qs2 = qs.values("patient_id").annotate(max_ids=Max("id"))
        qs = qs.filter(id__in=qs2.values("max_ids"))
        return qs


class MedWorkerListView(ListAPIView):
    """
    Возвращает список медработников
    """

    queryset = models.MedWorker.objects.all()
    serializer_class = ser.MedWorkerCommonSerializer
    filterset_class = filters.MedWorkerListFilter


# """Patients"""


class PatientAndCardCreateGeneric(CreateAPIView):
    """
    Регистрирует карту пациента и пациента для медработника с указанным id
    """

    serializer_class = ser.PatientAndCardSerializer

    def get_serializer_context(self):
        ret = super().get_serializer_context()
        ret["med_worker"] = models.MedWorker.objects.get(id=self.kwargs["id"])
        return ret

    # def get_permissions(self):
    #   return [IsAuthenticated()]
    #   # return []

    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)


class PatientAndCardUpdateView(mixins.RetrieveModelMixin, UpdateAPIView):
    """
    Обновление данных о пациенте и его конкретной карточки
    """

    serializer_class = ser.PatientAndCardSerializer
    lookup_url_kwarg = "id"

    def get_permissions(self):
        """Change permission for PUT and PATCH"""
        return super().get_permissions()

    def get_object(self):
        obj_id = self.kwargs.get(self.lookup_url_kwarg)
        obj = models.PatientCard.objects.select_related("patient").filter(
            id=obj_id
        )
        try:
            card = obj[0]
        except IndexError as er:
            raise Http404
        patient = card.patient
        ret = {"card": card, "patient": patient}
        return ret

    def get(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        return super().patch(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        kwargs["partial"] = True
        a = super().put(request, *args, **kwargs)
        return a


class PatientCardViewSet(ModelViewSet):
    serializer_class = ser.PatientCardDefaultSerializer
    queryset = models.PatientCard.objects.all()


class PatientShotsTableView(ListAPIView):
    """
    Информация о карточках пациента и если были снимки, то инфа о сниках (без самих снимков)
    """

    serializer_class = ser.PatientTableSerializer

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        ctx["patient"] = models.Patient.objects.filter(id=self.kwargs["id"])[0]
        return ctx

    def get_queryset(self):
        qs = (
            models.CTImage.objects.select_related(
                "ct_device", "patient_card", "image"
            )
            .prefetch_related(
                Prefetch(
                    "image__segments",
                    queryset=models.CTSegmentGroupInfo.objects.all(),
                )
            )
            .filter(patient_card__patient__id=self.kwargs["id"])
        )
        return qs

    def list(self, request, *args, **kwargs):
        try:
            l = super().list(request, *args, **kwargs)
            return l
        except IndexError:
            return Response(status=status.HTTP_404_NOT_FOUND)


class PatientListView(ListAPIView):
    """
    Список всех пациентов с возможностью фильтрации
    """

    queryset = models.Patient.objects.all()
    serializer_class = ser.PatientSerializer
    filterset_class = filters.PatientListFilter


# """CTs' views"""
class CTImageCreateView(CreateAPIView):
    """
    Форма для сохранния УЗИ изображения и отправки в очередь на обарботку
    УЗИ снимка
    """

    serializer_class = ser.CTImageCreateSerializer

    def create(self, request, *args, **kwargs):
        print(request.data["original_image"])
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.perform_create, serializer)
            data = future.result(timeout=300)
        headers = self.get_success_headers(serializer.data)
        return Response(data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer):
        d = serializer.save()

        ct_image: models.CTImage = d["ct_image"]
        original: models.OriginalImage = d["image"]
        task = tasks.send_prediction_task(
            original.image.tiff_file_path,
            ct_image.details.get("ct_type", "undefined"),
            ct_image.id,
        )
        result_backend = RedisBackend(host='localhost', port=6380)
        result = result_backend.get_result(message=task, block=True, timeout=1000000)
        return {"image_id": ct_image.id}


class CTImageShowView(RetrieveAPIView):
    """
    Информация об одной группе снимков
    """

    serializer_class = ser.CTImageGetSerializer

    def get_object(self):
        try:
            return self.get_queryset()[0]
        except IndexError as er:
            raise Http404

    def get_queryset(self):
        return (
            models.CTImage.objects.filter(id=self.kwargs["id"])
            .select_related("ct_device", "patient_card", "image")
            .prefetch_related(
                "patient_card__patient",
                "image__segments",
                "image__segments__data__points",
            )
        )


class CTOriginImageUpdateView(UpdateAPIView):
    """
    Обновление оригинального снимка (только параметры отображения)
    """

    queryset = models.OriginalImage.objects.all()
    serializer_class = ser.CTUpdateOriginalImageSerializer
    # permission_classes = [IsAuthenticated]
    lookup_url_kwarg = "id"


class CTDeviceView(ListAPIView):
    """
    Список аппаратов УЗИ
    """

    queryset = models.CTDevice.objects.all()
    serializer_class = ser.CTDeviceSerializer


class CTIdsView(ListAPIView):
    """
    Полученние даднных об узи по ид.
    TODO: добавить ручку на получениее данных у конкретного
    врача или всех узи.
    """

    serializer_class = ser.CTImageSupprotSerializer

    def get_queryset(self):
        ids = json.loads(self.request.query_params.get("ids", ""))
        return (
            models.CTImage.objects.filter(id__in=ids)
            .select_related("ct_device", "patient_card")
            .prefetch_related("patient_card__patient")
        )

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(self.list2dict(serializer.data))

        serializer = self.get_serializer(queryset, many=True)
        return Response(self.list2dict(serializer.data))

    def list2dict(self, data, lookup="id"):
        return {di[lookup]: di for di in data}


class CTShowUpdateView(UpdateAPIView):
    """
    Обновление всей страницы с информацией о приеме
    TODO: FIX 5 DB REQUESTS
    """

    serializer_class = ser.CTShowUpdateSerializer

    def get_object(self):
        try:
            return self.get_queryset()[0]
        except IndexError as er:
            raise Http404

    def get_queryset(self):
        return models.CTImage.objects.filter(
            id=self.kwargs["id"]
        ).select_related("patient_card")

    def put(self, request, *args, **kwargs):
        return super().put(request, *args, **kwargs)


class CTSegmentGroupListView(ListAPIView):
    filterset_class = filters.SegmentGroupFilter

    def get_queryset(self):
        qs = models.CTSegmentGroupInfo.objects.filter(
            original_image_id__in=models.CTImage.objects.filter(
                image=self.kwargs["ct_img_id"]
            ).values("image")
        )
        return qs

    def get_serializer_class(self):
        return ser.CTSegmentationGroupBaseSerializer


class CTSegmentGroupCreateView(CreateAPIView):
    serializer_class = ser.CTSegmentationGroupCreateSerializer


class CTSegmentGroupCreateSoloView(CreateAPIView):
    serializer_class = ser.CTSegmentationGroupCreateSoloSerializer


class CTSegmentGroupUpdateDeleteView(RetrieveUpdateDestroyAPIView):
    serializer_class = ser.CTSegmentationGroupUpdateDeleteSerializer
    lookup_url_kwarg = "id"
    lookup_field = "pk"

    def get_queryset(self):
        qs = (
            models.CTSegmentGroupInfo.objects.filter(id=self.kwargs["id"])
            # .prefetch_related('points')
        )
        return qs

    def get_object(self):
        return super().get_object()


class CTSegmentAddView(CreateAPIView):
    serializer_class = ser.CTSegmentationAddSerializer


class CTSegmentUpdateDeleteView(RetrieveUpdateDestroyAPIView):
    serializer_class = ser.CTSegmentationUpdateDeleteSerializer
    lookup_url_kwarg = "id"
    lookup_field = "pk"

    def get_queryset(self):
        qs = models.SegmentationData.objects.filter(
            id=self.kwargs["id"]
        ).prefetch_related("points")
        return qs


class CTSegmentCopyView(RetrieveAPIView):
    serializer_class = ser.CTImageGetSerializer

    def get_object(self):
        try:
            return self.get_queryset()[0]
        except IndexError as er:
            raise Http404

    def get_queryset(self):
        return (
            models.CTImage.objects.filter(id=self.kwargs["id"])
            .select_related("ct_device", "patient_card", "image")
            .prefetch_related(
                "patient_card__patient",
                "image__segments",
                "image__segments__data__points",
            )
        )
