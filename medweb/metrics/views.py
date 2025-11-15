from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.db.models import Sum
from medml.models import OriginalImage


def index(request: HttpRequest) -> HttpResponse:
    count = {"orig": 0, "viewed": 0}

    count["orig"] = OriginalImage.objects.count()
    count["viewed"] = OriginalImage.objects.filter(viewed_flag=True).count()

    data = (
        'Images_Count{label="Original"} '
        + str(count["orig"])
        + '\nImages_Count{label="Processed"} '
        + str(count["viewed"])
        + "\nImages_In_Process_Count "
        + str(count["orig"] - count["viewed"])
    )

    return HttpResponse(data, content_type="text/plain")
