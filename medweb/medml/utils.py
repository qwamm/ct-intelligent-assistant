from django.conf import settings
from datetime import date
import hashlib
import os
import tifffile as tiff
import pydicom
from tempfile import NamedTemporaryFile


"""
  See docs, why dynamic not working
  https://docs.djangoproject.com/en/4.1/topics/migrations/#migration-serializing
  Any function or method reference (e.g. datetime.datetime.today) (must be in module’s top-level scope)


  class UploadPathConstructor:
    def __init__(self, basedir: str) -> None:
      self.basedir: str = basedir

    def uzi_upload_path(self, obj, filename: str):
      filen, filee = os.path.splitext(filename)
      h = hashlib.sha256(filen).hexdigest()[:settings.IMAGE_NAME_MAX_CHARS]
      return f"{self.basedir}/%Y/{obj.patient.id}/{h}.{filee}"
"""


def _makePath(obj, filename: str, basedir: str):
    filen, filee = os.path.splitext(filename)
    filee = filee.lower()
    h = hashlib.sha256(filen.encode("utf-8")).hexdigest()[
        : settings.IMAGE_NAME_MAX_CHARS
    ]
    if filee in {".jpeg", ".jpg"}:
        filee = ".png"
    print(f"{basedir}/{date.today().year}/{h}{filee}")
    return f"{basedir}/{date.today().year}/{h}{filee}"


def originalUZIPath(obj, filename: str):
    return _makePath(obj, filename, "originalUZI")


def mlModelPath(obj, filename: str):
    filen, filee = os.path.splitext(filename)
    h = hashlib.sha256(filen.encode("utf-8")).hexdigest()[
        : settings.IMAGE_NAME_MAX_CHARS
    ]

    return f"nnModel/{date.today().year}/{h}{filee}"


def getFields(obj, has_id=True, add_name=""):
    d = {f"{add_name}{f.name}": getattr(obj, f.name) for f in obj._meta.fields}
    if not has_id:
        d.pop(f"{add_name}id")
    return d


def updateClassesToGroup(classes, group):
    for key in classes:
        group.details[f"nodule_{key}"] = classes[key]
    group.details["nodule_type"] = int(
        max(classes.items(), key=lambda x: x[1])[0]
    )


def in_mem_image_pre_saver(image):
    fbase, filee = os.path.splitext(image.name)
    filee = filee.lower()
    if filee in {".tif", ".tiff"}:
        # fixing tiff images
        tmp = NamedTemporaryFile()
        try:
            with image.open() as inp:
                t = tiff.imread(inp, squeeze=False)
            tiff.imwrite(tmp, t, compression=tiff.COMPRESSION.PNG)
            image.file = tmp
        except tiff.TiffFileError as e:
            raise AttributeError(f"Битый .tif файл. {e}")
        return image, t.shape[0]
    elif filee in {".dcm"}:
        # convert .dcm to tiff
        tmp = NamedTemporaryFile()
        try:
            with image.open() as inp:
                t = pydicom.dcmread(inp)
            img_size = t.pixel_array.shape[0]
            tiff.imwrite(tmp, t.pixel_array, compression=tiff.COMPRESSION.PNG)
            image.file = tmp
            image.name = fbase + ".tif"
        except Exception as e:
            print(e)
            raise AttributeError(f"Битый .dcm файл.")
        return image, img_size
    return image, 1
