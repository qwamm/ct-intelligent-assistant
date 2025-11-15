from django.contrib import admin

# Register your models here.
from inner_mail import models


@admin.register(models.MailDetails)
class MailDetailsAdmin(admin.ModelAdmin):
    pass


@admin.register(models.NotificationGroup)
class NotificationGroupAdmin(admin.ModelAdmin):
    pass


@admin.register(models.Notification)
class NotificationAdmin(admin.ModelAdmin):
    pass


@admin.register(models.NotificationDynamics)
class NotificationDynamicsAdmin(admin.ModelAdmin):
    list_display = ("mail", "user", "status")
