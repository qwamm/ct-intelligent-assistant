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

from medml import filters
from medml import serializers as ser
from medml import models
from medml import tasks

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
            models.UZIImage.objects.select_related(
                "uzi_device", "patient_card", "image"
            )
            .prefetch_related(
                Prefetch(
                    "image__segments",
                    queryset=models.UZISegmentGroupInfo.objects.all(),
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


# """UZIs' views"""
class UZIImageCreateView(CreateAPIView):
    """
    Форма для сохранния УЗИ изображения и отправки в очередь на обарботку
    УЗИ снимка
    """

    serializer_class = ser.UZIImageCreateSerializer

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

        uzi_image: models.UZIImage = d["uzi_image"]
        original: models.OriginalImage = d["image"]
        task = tasks.send_prediction_task(
            original.image.tiff_file_path,
            uzi_image.details.get("projection_type", "cross"),
            uzi_image.id,
        )
        result_backend = RedisBackend(host='localhost', port=6380)
        result = result_backend.get_result(message=task, block=True, timeout=1000000)
        return {"image_id": uzi_image.id}


class UziImageShowView(RetrieveAPIView):
    """
    Информация об одной группе снимков
    """

    serializer_class = ser.UZIImageGetSerializer

    def get_object(self):
        try:
            return self.get_queryset()[0]
        except IndexError as er:
            raise Http404

    def get_queryset(self):
        return (
            models.UZIImage.objects.filter(id=self.kwargs["id"])
            .select_related("uzi_device", "patient_card", "image")
            .prefetch_related(
                "patient_card__patient",
                "image__segments",
                "image__segments__data__points",
            )
        )


class UZIOriginImageUpdateView(UpdateAPIView):
    """
    Обновление оригинального снимка (только параметры отображения)
    """

    queryset = models.OriginalImage.objects.all()
    serializer_class = ser.UZIUpdateOriginalImageSerializer
    # permission_classes = [IsAuthenticated]
    lookup_url_kwarg = "id"


class UZIDeviceView(ListAPIView):
    """
    Список аппаратов УЗИ
    """

    queryset = models.UZIDevice.objects.all()
    serializer_class = ser.UZIDeviceSerializer


class UZIIdsView(ListAPIView):
    """
    Полученние даднных об узи по ид.
    TODO: добавить ручку на получениее данных у конкретного
    врача или всех узи.
    """

    serializer_class = ser.UZIImageSupprotSerializer

    def get_queryset(self):
        ids = json.loads(self.request.query_params.get("ids", ""))
        return (
            models.UZIImage.objects.filter(id__in=ids)
            .select_related("uzi_device", "patient_card")
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


class UZIShowUpdateView(UpdateAPIView):
    """
    Обновление всей страницы с информацией о приеме
    TODO: FIX 5 DB REQUESTS
    """

    serializer_class = ser.UZIShowUpdateSerializer

    def get_object(self):
        try:
            return self.get_queryset()[0]
        except IndexError as er:
            raise Http404

    def get_queryset(self):
        return models.UZIImage.objects.filter(
            id=self.kwargs["id"]
        ).select_related("patient_card")

    def put(self, request, *args, **kwargs):
        return super().put(request, *args, **kwargs)


class ModelUpdateView(CreateAPIView):
    """
    Обновление весов конкретной модели
    """

    serializer_class = ser.MLModelSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        qs = self.get_queryset()
        try:
            seg = qs[0]
            return seg
        except:
            return None

    def get_queryset(self):
        qs = models.MLModel.objects.filter(id=self.kwargs["id"])
        return qs

    def post(self, request: Request, *args, **kwargs):
        if request.POST:
            nnmodel = self.get_object()
            if nnmodel:
                tasks.update_model_weights.delay(  # TODO: добавить таску и названия весов по умолчанию
                    nnmodel.file.path,
                    nnmodel.model_type,
                    nnmodel.projection_type,
                )

        return Response(status=status.HTTP_201_CREATED)


class UZISegmentGroupListView(ListAPIView):
    filterset_class = filters.SegmentGroupFilter

    def get_queryset(self):
        qs = models.UZISegmentGroupInfo.objects.filter(
            original_image_id__in=models.UZIImage.objects.filter(
                image=self.kwargs["uzi_img_id"]
            ).values("image")
        )
        return qs

    def get_serializer_class(self):
        return ser.UZISegmentationGroupBaseSerializer


class UZISegmentGroupCreateView(CreateAPIView):
    serializer_class = ser.UZISegmentationGroupCreateSerializer


class UZISegmentGroupCreateSoloView(CreateAPIView):
    serializer_class = ser.UZISegmentationGroupCreateSoloSerializer


class UZISegmentGroupUpdateDeleteView(RetrieveUpdateDestroyAPIView):
    serializer_class = ser.UZISegmentationGroupUpdateDeleteSerializer
    lookup_url_kwarg = "id"
    lookup_field = "pk"

    def get_queryset(self):
        qs = (
            models.UZISegmentGroupInfo.objects.filter(id=self.kwargs["id"])
            # .prefetch_related('points')
        )
        return qs

    def get_object(self):
        return super().get_object()


class UZISegmentAddView(CreateAPIView):
    serializer_class = ser.UZISegmentationAddSerializer


class UZISegmentUpdateDeleteView(RetrieveUpdateDestroyAPIView):
    serializer_class = ser.UZISegmentationUpdateDeleteSerializer
    lookup_url_kwarg = "id"
    lookup_field = "pk"

    def get_queryset(self):
        qs = models.SegmentationData.objects.filter(
            id=self.kwargs["id"]
        ).prefetch_related("points")
        return qs


class UziSegmentCopyView(RetrieveAPIView):
    serializer_class = ser.UZIImageGetSerializer

    def get_object(self):
        try:
            return self.get_queryset()[0]
        except IndexError as er:
            raise Http404

    def get_queryset(self):
        return (
            models.UZIImage.objects.filter(id=self.kwargs["id"])
            .select_related("uzi_device", "patient_card", "image")
            .prefetch_related(
                "patient_card__patient",
                "image__segments",
                "image__segments__data__points",
            )
        )
