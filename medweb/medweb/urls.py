from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

# import django_prometheus


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v3/", include("medml.urls")),
    path("api/v3/inner_mail/", include("inner_mail.urls")),
    path("api/v3/metrics/", include("metrics.urls")),
    path("api/prometheus/v1/", include("django_prometheus.urls")),
    # path('', include('medweb_front.urls')),
]

urlpatterns += static(
    settings.STATIC_URL, document_root=settings.STATIC_ROOT
)
urlpatterns += static(
    settings.MEDIA_URL, document_root=settings.MEDIA_ROOT
)

if settings.DEBUG:
    # https://django-debug-toolbar.readthedocs.io/en/latest/installation.html
    # https://drf-yasg.readthedocs.io/en/stable/readme.html
    from django.urls import re_path
    from rest_framework import permissions
    from drf_yasg.views import get_schema_view
    from drf_yasg import openapi

    schema_view = get_schema_view(
        openapi.Info(
            title="MedML API",
            default_version="v3",
            description="Test description",
            contact=openapi.Contact(email="newmancu@gmail.com"),
        ),
        public=True,
        permission_classes=[permissions.AllowAny],
    )

    urlpatterns += [
        path(
            "api/docs/",
            include(
                [
                    re_path(
                        r"^swagger(?P<format>\.json|\.yaml)$",
                        schema_view.without_ui(cache_timeout=0),
                        name="schema-json",
                    ),
                    re_path(
                        r"^swagger/$",
                        schema_view.with_ui("swagger", cache_timeout=0),
                        name="schema-swagger-ui",
                    ),
                    re_path(
                        r"^redoc/$",
                        schema_view.with_ui("redoc", cache_timeout=0),
                        name="schema-redoc",
                    ),
                ]
            ),
        )
    ]
    urlpatterns += [path("__debug__/", include("debug_toolbar.urls"))]
