import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from . import cityscapes


def get_images(
    model_base_path="/home/azureml-model",
    model_name="deeplab_v3plus_256",
    dataset_path="/azureml-blobstore/cityscapes",
    image_id=None,
):

    model = tf.keras.models.load_model(
        Path(model_base_path, model_name, "model/data/model"),
        custom_objects={
            "UpdatedMeanIoU": cityscapes.UpdatedMeanIoU,
            "jaccard_loss": cityscapes.jaccard_loss,
        },
    )

    resize = int(model_name.replace("_augment", "").split("_")[-1])
    img_size = (resize, resize)

    leftImg8bit_path = Path(dataset_path, "leftImg8bit")
    gtFine_path = Path(dataset_path, "gtFine")

    image_id = str(image_id)
    input_img_paths = []
    labels_img_paths = []
    if image_id:
        input_img_paths = sorted(
            Path(leftImg8bit_path).glob(f"**/*{image_id}*_leftImg8bit.png")
        )
        labels_img_paths = sorted(
            Path(gtFine_path).glob(f"**/*{image_id}*_color.png")
        )

    if len(input_img_paths) == 0 or len(labels_img_paths) == 0:
        input_img_paths = sorted(
            Path(leftImg8bit_path, "val").glob("**/*_leftImg8bit.png")
        )
        labels_img_paths = sorted(
            Path(gtFine_path, "val").glob("**/*_color.png")
        )

    rand_idx = np.random.randint(0, len(input_img_paths))

    with open(input_img_paths[rand_idx], "rb") as f:
        original_img_b64 = base64.b64encode(f.read())
        original_img_b64_str = original_img_b64.decode("utf-8")

        input_img = Image.open(
            BytesIO(base64.b64decode(original_img_b64))
        ).resize(img_size)
        categories_img = Image.fromarray(
            cityscapes.cityscapes_category_ids_to_category_colors(
                np.squeeze(
                    np.argmax(
                        model.predict(np.expand_dims(input_img, 0)), axis=-1
                    )
                )
            )
        )

        buffered = BytesIO()
        categories_img.save(buffered, format="PNG")
        categories_img_b64_str = base64.b64encode(buffered.getvalue()).decode(
            "utf-8"
        )

    with open(labels_img_paths[rand_idx], "rb") as f:
        labels_img_read = f.read()
        labels_img_b64 = base64.b64encode(labels_img_read)
        labels_img_b64_str = labels_img_b64.decode("utf-8")

    return original_img_b64_str, labels_img_b64_str, categories_img_b64_str
