from django.urls import path, include
from metrics import views


urlpatterns = [path("metrics/", views.index, name="metrics")]
