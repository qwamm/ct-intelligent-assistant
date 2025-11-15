from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from django.test.testcases import SerializeMixin
from medml import models


class MedWorkerTestCaseMixin(SerializeMixin):
    lockfile = __file__

    USER1_DATA = {
        "email": "superUser1@gmail.com",
        "last_name": "User",
        "first_name": "Super",
        "fathers_name": "Olegovich",
        "med_organization": "Moscow policy",
        "password1": "superSecret1",
        "password2": "superSecret1",
    }


class MedWorkerTests(MedWorkerTestCaseMixin, APITestCase):
    def TestMedworkerRegistr(self):
        url = reverse("registration")
        response = self.client.post(url, self.USER1_DATA, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(models.MedWorker.objects.count(), 1)
        self.assertEqual(
            models.MedWorker.objects.get(email=self.USER1_DATA["email"]).email,
            self.USER1_DATA["email"],
        )

    def TestMedworkerLogin(self):
        url = reverse("token_obtain_pair")
        login_data = {
            "email": self.USER1_DATA["email"],
            "password": self.USER1_DATA["password1"],
        }
        response = self.client.post(url, login_data, format="json")
        self.assertEqual(
            models.MedWorker.objects.get(email=self.USER1_DATA["email"]).email,
            self.USER1_DATA["email"],
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.USER_TOKEN = response.json()

    def TestMedWorkerRefreshToken(self):
        url = reverse("token_refresh")
        refresh_data = {"refresh": self.USER_TOKEN["refresh"]}
        response = self.client.post(url, refresh_data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse("refresh" in response.json())
        self.assertTrue("access" in response.json())

    def TestMedWorkerUpdate(self):
        url = reverse("med_worker_update")
        profile_data = self.USER1_DATA.copy()
        profile_data.pop("password1")
        profile_data.pop("password2")
        profile_data.pop("email")
        profile_data.update({"job": None, "id": 1})

        self.client.login(
            email=self.USER1_DATA["email"],
            password=self.USER1_DATA["password1"],
        )

        # get test
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.json(), profile_data)

        # update test
        update_data = {"job": "Super job"}
        res = self.client.patch(url, update_data, format="json")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(
            models.MedWorker.objects.get(email=self.USER1_DATA["email"]).job,
            update_data["job"],
        )

    def test_sequal(self):
        self.TestMedworkerRegistr()
        self.TestMedworkerLogin()
        self.TestMedWorkerRefreshToken()
        self.TestMedWorkerUpdate()


class PatientTests(MedWorkerTestCaseMixin, APITestCase):
    PATIENT_DATA = {
        "patient": {
            "first_name": "Игроь",
            "last_name": "Горохов",
            "fathers_name": "Игорьевич",
            "personal_policy": "1234123412341234",
            "email": "Igor.cal@email.or",
            "is_active": True,
        },
        "card": {
            "has_nodules": models.PatientCard.NODULES_CHOICES[0][0],
            "diagnosis": "Что-то обнаржулили площадью 290",
        },
    }

    def setUp(self) -> None:
        udata = self.USER1_DATA.copy()
        udata["password"] = udata.pop("password1")
        udata.pop("password2")
        user = models.MedWorker.objects.create(**udata)
        self.client.force_login(user)

    def TestCreatePatient(self):
        url = reverse("patient_create")

        response = self.client.post(url, self.PATIENT_DATA, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            models.Patient.objects.get(
                email=self.PATIENT_DATA["patient"]["email"]
            ).email,
            self.PATIENT_DATA["patient"]["email"],
        )
        self.assertEqual(
            models.PatientCard.objects.get(id=1).diagnosis,
            self.PATIENT_DATA["card"]["diagnosis"],
        )

    def TestUpdatePatient(self):
        url = reverse("patient_update", args=(1,))

        # get
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        rdata = response.json()
        self.assertEqual(rdata["patient"]["id"], 1)
        self.assertEqual(rdata["card"]["id"], 1)
        rdata["patient"].pop("id")
        rdata["card"].pop("id")
        rdata["card"].pop("acceptance_datetime")
        self.assertEqual(rdata, self.PATIENT_DATA)

        # update
        patch_data = {"patient": {"is_active": False}, "card": {}}
        res = self.client.patch(url, patch_data, format="json")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertFalse(models.Patient.objects.get(id=1).is_active)

    def test_sequal(self):
        self.TestCreatePatient()
        self.TestUpdatePatient()
