from django.urls import path, include
from inner_mail import views


urlpatterns = [
    path(
        "notifications/",
        include(
            [
                # Получение всех уведомлений указанного пользователя (+ фильтрация по статусу)
                path(
                    "all/<int:pk>/",
                    views.NotificationDynamicsGenerics.as_view(),
                    name="new_notifications",
                ),
                # Изменения уведомления на прочитанное
                path(
                    "mark/viewed/",
                    views.NotificationDynamicsViewedGenerics.as_view(),
                    name="mark_viewed",
                ),
                # Получение всех уведомлений в группе уведомлений (+ проверка на состояние пользователя в группе)
                path(
                    "group/<int:pk>/",
                    views.NotificationsGroupGenerics.as_view(),
                    name="notifications_group",
                ),
                # Получение списка группы, в которых состоит пользователь
                path(
                    "groups/",
                    views.NotificationsGroupListGenerics.as_view(),
                    name="notifications_groups",
                ),
                # Создать уведомление
                path(
                    "create/",
                    views.NotificationDynamicsCreateGenerics.as_view(),
                    name="create",
                ),
                # Отправка ответа на уведомление
                path(
                    "reply/",
                    views.NotificationDynamicsReplyGenerics.as_view(),
                    name="reply",
                ),
            ]
        ),
    ),
    # Создать сообщение
    path(
        "mail/create/",
        include(
            [
                path(
                    "simple/",
                    views.SimpleMailCreateGenerics.as_view(),
                    name="simple_mail",
                ),
                path(
                    "expert/",
                    views.ExpertMailCreateGenerics.as_view(),
                    name="expert_mail",
                ),
            ]
        ),
    ),
]
