# Приложение "Интеллектуальный ассистент УЗИ щитовидной железы"

### 🖥️ Локальное развёртывание

**Требования:**
- Python 3.11+
- PostgreSQL 16.0+
- React 19.1.1+
- npm 10.9.3

#### Предварительные действия:
  - Установить клиент pgAdmin(если его нет) и создать БД с именем dev_db, работующую на сокете  ```localhost:5432 ```

#### Запуск компонентов(все команды, кроме последней, выполняются из корня проекта):
   ```bash
   pip install -r requirements.txt
   python medweb/manage.py makemigrations medml
   python medweb/manage.py makemigrations inner_mail
   python medweb/manage.py migrate
   python dj_nnapi/dj_nnapi/manage.py makemigrations nnmodel
   python dj_nnapi/dj_nnapi/manage.py migrate
   python medweb/manage.py base_configuration # скрипт для первичного заполнения БД данными об УЗИ аппаратах и аккаунте администратора
   python medweb/manage.py runserver #запуск основного сервера, работающего на порте 8000
   сd dj_nnapi/dj_nnapi
   python manage.py rundramatiq --processes 1 --threads 1 -v 2 --queues predict_all #запуск внешнего REST сервера, реализующего апи для работы с моделями через daranatiq
   ```
