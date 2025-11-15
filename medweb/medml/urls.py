from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from rest_framework import routers
from django.urls import path, include

from medml import views

router = routers.DefaultRouter()
router.register("", views.PatientCardViewSet)


urlpatterns = [
    path(
        "auth/register/", views.RegistrationView.as_view(), name="registration"
    ),
    path(
        "auth/login/", TokenObtainPairView.as_view(), name="token_obtain_pair"
    ),
    path(
        "auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"
    ),
    path(
        "model/update/<int:id>",
        views.ModelUpdateView.as_view(),
        name="model_update",
    ),
    path(
        "med_worker/",
        include(
            [
                path(
                    "list/",
                    views.MedWorkerListView.as_view(),
                    name="med_worker_list_view",
                ),
                # Изменить информацию о мед работнике
                path(
                    "update/<int:id>",
                    views.MedWorkerChangeView.as_view(),
                    name="med_worker_update",
                ),
                # Для Start Page - инфа только о последенй карточке
                path(
                    "patients/<int:id>",
                    views.MedWorkerPatientsTableView.as_view(),
                    name="med_worker_patients_table",
                ),
            ]
        ),
    ),
    path(
        "patient/",
        include(
            [
                path(
                    "list/",
                    views.PatientListView.as_view(),
                    name="patient_list_view",
                ),
                # Создание пациента и карточки
                path(
                    "create/<int:id>",
                    views.PatientAndCardCreateGeneric.as_view(),
                    name="patient_create",
                ),
                # Изменение только информации о пациенте и его конкретной карточки
                path(
                    "update/<int:id>/",
                    views.PatientAndCardUpdateView.as_view(),
                    name="patient_update",
                ),
                # инфа о карточках пациента и если были снимки, то инфа о сниках (без самих снимков)
                path(
                    "shots/<int:id>/",
                    views.PatientShotsTableView.as_view(),
                    name="patient_shots_table",
                ),
            ]
        ),
    ),
    path(
        "uzi/",
        include(
            [
                # Получить список исследований УЗИ по ид
                path("ids/", views.UZIIdsView.as_view(), name="uzi_ids"),
                # Список аппаратов УЗИ
                path(
                    "devices/",
                    views.UZIDeviceView.as_view(),
                    name="uzi_devices",
                ),
                # Информация об одной группе снимков
                path(
                    "<int:id>/",
                    views.UziImageShowView.as_view(),
                    name="uzi_image_show",
                ),
                # Обновление всей страницы с информацией о приеме
                path(
                    "<int:id>/update/",
                    views.UZIShowUpdateView.as_view(),
                    name="uzi_show_update",
                ),
                # Обновление оригинального снимка (только параметры отображения)
                path(
                    "update/origin/<int:id>",
                    views.UZIOriginImageUpdateView.as_view(),
                    name="uzi_update_origin",
                ),
                # Создать Группу изображений и добавить снимок оригинала
                path(
                    "create/",
                    views.UZIImageCreateView.as_view(),
                    name="uzi_group_create",
                ),
                path(
                    "segment/",
                    include(
                        [
                            path(
                                "group/<int:uzi_img_id>/",
                                views.UZISegmentGroupListView.as_view(),
                                name="uzi_segment_group_list",
                            ),
                            path(
                                "group/create/<int:uzi_img_id>/",
                                views.UZISegmentGroupCreateView.as_view(),
                                name="uzi_segment_group_create",
                            ),
                            path(
                                "group/create/solo/<int:uzi_img_id>/",
                                views.UZISegmentGroupCreateSoloView.as_view(),
                                name="uzi_segment_group_create_solo",
                            ),
                            path(
                                "group/update/<int:id>/",
                                views.UZISegmentGroupUpdateDeleteView.as_view(),
                                name="uzi_segment_group_update_delete",
                            ),
                            path(
                                "add/",
                                views.UZISegmentAddView.as_view(),
                                name="uzi_segment_add",
                            ),
                            path(
                                "update/<int:id>/",
                                views.UZISegmentUpdateDeleteView.as_view(),
                                name="uzi_segment_update_delete",
                            ),
                            path(
                                "copy/<int:id>/",
                                views.UziSegmentCopyView.as_view(),
                                name="uzi_segment_copy",
                            ),
                            # path()
                            # # Получение/Добавление сегментов на изображении
                            # path('<int:uzi_img_id>/', views.UZISegmentGetCreateView.as_view(), name='uzi_segment_get_create'),
                            # # Удаление/Изменение ифнормации о сегменте
                            # path('<int:img_id>/<int:data_id>', views.UZISegmentDeleteUpdateView.as_view(), name='uzi_segment_delete_update'),
                        ]
                    ),
                ),
            ]
        ),
    ),
    path("card/", include(router.urls)),
]
