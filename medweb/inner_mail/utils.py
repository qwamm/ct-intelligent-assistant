from inner_mail import models
from medml import models as med_models


def get_details(details: dict):
    if details["mail_type"] == models.MailType.MSG:
        details["card"] = None
    return models.MailDetails.objects.create(**details)
    # raise AttributeError("NO SUCH TYPE")
