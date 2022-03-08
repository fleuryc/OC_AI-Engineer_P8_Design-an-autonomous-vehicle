import base64
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
from inference_schema.schema_decorators import input_schema, output_schema
from PIL import Image

import cityscapes


def init():
    global model

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/data/model"
    )
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "UpdatedMeanIoU": cityscapes.UpdatedMeanIoU,
            "jaccard_loss": cityscapes.jaccard_loss,
        },
    )


# Automatically generate a Swagger schema
# see : https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script#automatically-generate-a-swagger-schema
input_sample = "<<base64 encoded png image>>"
output_sample = "<<base64 encoded mask>>"


@input_schema(
    "image",
    StandardPythonParameterType(input_sample),
)
@output_schema(
    StandardPythonParameterType(output_sample),
)
def run(image):
    # infer img size from model name
    resize = int(model.name.replace("_augment", "").split("_")[-1])
    img_size = (resize, resize)

    input_img = Image.open(BytesIO(base64.b64decode(image))).resize(img_size)

    buffered = BytesIO()
    Image.fromarray(
        cityscapes.cityscapes_category_ids_to_category_colors(
            np.squeeze(
                np.argmax(
                    model.predict(np.expand_dims(input_img, 0)),
                    axis=-1,
                )
            )
        )
    ).save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")
