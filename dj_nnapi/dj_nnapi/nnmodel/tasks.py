import uuid

import dramatiq
from nnmodel import models

from dj_nnapi.dj_nnapi.nnmodel.forms import segmetationDataForm
import cv2
import numpy as np

from dj_nnapi.dj_nnapi.nnmodel.nn.CTModel import CTModel


def createUziSegmentGroup(details, uzi_image_id, form=segmetationDataForm):
    return models.UZISegmentGroupInfo(
        details=details, is_ai=True, original_image_id=uzi_image_id
    )

def createSegmentationDataObj(ind, nodule, details, image_id, result_masks):
    match nodule:
        case 'TIRADS1':
            nodule_type = 1
        case 'TIRADS2':
            nodule_type = 2
        case 'TIRADS3':
            nodule_type = 3
        case 'TIRADS4':
            nodule_type = 4
        case 'TIRADS5':
            nodule_type = 5
        case _:
            nodule_type = -1

    nodule_2_3 = 1 if nodule_type == 2 or nodule_type == 3 else 0
    nodule_4 = 1 if nodule_type == 4 else 0
    nodule_5 = 1 if nodule_type == 5 else 0

    pre_details = {'nodule_type': nodule_type, 'nodule_2_3': nodule_2_3, 'nodule_4': nodule_4, 'nodule_5': nodule_5}

    details[ind] = createUziSegmentGroup(pre_details, image_id)
    models.UZISegmentGroupInfo.objects.bulk_create([details[ind]])
    segmentation_data_obj = models.SegmentationData(
        segment_group=details[ind],
        details=pre_details
    )
    segmentation_data_obj.save()
    createSegmentationPointObj(result_masks, segmentation_data_obj)

def createSegmentationPointObj(result_masks, segmentation_data_obj):
    segments_points = []
    # Проходим по всем маскам результатов
    for mask_idx, result_mask in enumerate(result_masks):
        if result_mask is None or np.sum(result_mask) == 0:
            print(f"Mask {mask_idx} is empty, skipping...")
            continue

        print(f"Processing mask {mask_idx} with shape {result_mask.shape}")

        # Преобразуем маску в uint8 для OpenCV
        binary_mask = (result_mask * 255).astype(np.uint8)

        # Находим контуры
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE | cv2.CHAIN_APPROX_TC89_L1,
        )

        print(f"Found {len(contours)} contours in mask {mask_idx}")

        # Обрабатываем найденные контуры
        for contour_idx, contour in enumerate(contours):
            print(f"Contour {contour_idx} has {len(contour)} points")

            # Создаем точки для каждого контура
            for point_idx, point in enumerate(contour):
                x = point[0][0]
                y = point[0][1]

                segments_points.append(
                    models.SegmentationPoint(
                        uid=hash(uuid.uuid4()),
                        segment=segmentation_data_obj,
                        x=x,
                        y=y,
                        z=mask_idx,
                    )
                )

    print(f"Total points to create: {len(segments_points)}")

    # Сохраняем точки сегментации батчами
    if segments_points:
        batch_size = 1000
        for i in range(0, len(segments_points), batch_size):
            batch = segments_points[i:i + batch_size]
            models.SegmentationPoint.objects.bulk_create(batch)
        print(f"Successfully created {len(segments_points)} segmentation points")
    else:
        print("No segmentation points to create")

@dramatiq.actor(queue_name='predict_all', store_results=True)
def predict_all(file_path: str, projection_type: str, id: int):
    predict_test = CTModel()
    predict_test.load()
    predict_test.predict(file_path)

    details = {}

    if isinstance(nodule_class_dict, dict):
        for ind, nodule in nodule_class_dict.items():
            createSegmentationDataObj(ind, nodule, details, id, result_masks)
    elif isinstance(nodule_class_dict, str):
        createSegmentationDataObj(0, nodule_class_dict, details, id, result_masks)
    else:
        raise TypeError("Unexpected type in nodule_class_dict")

    models.OriginalImage.objects.filter(id=id).update(viewed_flag=True)
    print("predicted!")
    return