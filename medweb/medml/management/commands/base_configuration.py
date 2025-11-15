from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from medml.models import MedWorker, PatientCard, Patient, UZIDevice, UZIImage


class Command(BaseCommand):
    help = "startup command"

    def handle(self, *args, **options):
        try:
            medworker = MedWorker.objects.get(email="admin@admin.ad")
        except:
            medworker = MedWorker.objects.create_superuser(
                "admin@admin.ad",
                "admin",
                first_name="admin",
                last_name="admin",
                fathers_name="admin",
            )
            self.stdout.write(self.style.SUCCESS(f"adding default superuser"))

        try:
            patient = Patient.objects.get(id=1)
        except:
            patient = Patient.objects.create(
                first_name="Иван",
                last_name="Иванов",
                fathers_name="Иванович",
                personal_policy="1234123412341234",
                email="iii@medml.med",
            )

        try:
            pc = PatientCard.objects.get(id=1)
        except:
            pc = PatientCard.objects.create(
                patient=patient, med_worker=medworker
            )

        try:
            uzid = UZIDevice.objects.get(id=1)
        except:
            uzid = UZIDevice.objects.create(name="GE Voluson E8")
            UZIDevice.objects.create(name="Logiq E9")

        self._create_groups()
        self._create_demo()

    def _create_groups(self, *args, **kwargs):
        name = "Patient"
        try:
            ex_group = Group.objects.get(name=name)
            self.stdout.write(self.style.WARNING(f"Group '{name}' exsists"))
            return
        except:
            pass

        ct = ContentType.objects.create(app_label="medml", model="MedWorker")
        permisssion = Permission.objects.create(
            codename=name.lower(), name=name, content_type=ct
        )

        group = Group.objects.create(name=name)
        Group.permissions.through.objects.create(
            group_id=group.id, permission_id=permisssion.id
        )
        self.stdout.write(self.style.SUCCESS("Creating Base Patient Rules"))
        pkws = {
            "first_name": "Петор",
            "last_name": "Петров",
            "fathers_name": "Петрович",
            "email": "patient@base.email",
        }
        patient_user = MedWorker.objects.create_user(
            password="patient_base", **pkws
        )
        patient_pat = Patient.objects.create(
            personal_policy="4321432143124321", **pkws
        )
        try:
            pc = PatientCard.objects.get(id=2)
        except:
            pc = PatientCard.objects.create(
                patient=patient_pat, med_worker=patient_user
            )
        patient_user.groups.add(group)
        self.stdout.write(self.style.SUCCESS("Base Patient User Were Created"))

    def _create_demo(self, *args, **kwargs):
        try:
            medworker = MedWorker.objects.get(email="demo@demo.com")
        except:
            medworker = MedWorker.objects.create_superuser(
                "demo@demo.com",
                "demopassword",
                first_name="demo",
                last_name="demo",
                fathers_name="demo",
            )
            self.stdout.write(self.style.SUCCESS(f"adding default superuser"))
