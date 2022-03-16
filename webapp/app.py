import os
from pathlib import Path

from azureml.core import Model, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from flask import Flask, jsonify, render_template, request

import src.utils as utils

DATASET_PATH = "/azureml-blobstore/cityscapes"
MODEL_PATH = "/home/azureml-model"
MODEL_NAME = "deeplab_v3plus_256"


app = Flask(__name__)


if not Path(MODEL_PATH, MODEL_NAME, "model/data/model").exists():
    sp = ServicePrincipalAuthentication(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        service_principal_id=os.environ["AZURE_SERVICE_PRINCIPAL_ID"],
        service_principal_password=os.environ[
            "AZURE_SERVICE_PRINCIPAL_PASSWORD"
        ],
    )
    ws = Workspace(
        auth=sp,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )
    model = Model(ws, MODEL_NAME)
    model.download(target_dir=str(Path(MODEL_PATH, MODEL_NAME)))


@app.route("/")
@app.route("/api")
def index(image_id=""):
    original_img_b64_str, labels_img_b64_str, categories_img_b64_str = (
        None,
        None,
        None,
    )

    if request.args.get("image_id"):
        image_id = request.args.get("image_id")
        (
            original_img_b64_str,
            labels_img_b64_str,
            categories_img_b64_str,
        ) = utils.get_images(
            model_base_path=MODEL_PATH,
            model_name=MODEL_NAME,
            dataset_path=DATASET_PATH,
            image_id=image_id,
        )

    if request.path == "/api":
        return jsonify(
            original_img_b64_str=original_img_b64_str,
            labels_img_b64_str=labels_img_b64_str,
            categories_img_b64_str=categories_img_b64_str,
        )

    return render_template(
        "index.html",
        image_id=image_id,
        original_img_b64_str=original_img_b64_str,
        labels_img_b64_str=labels_img_b64_str,
        categories_img_b64_str=categories_img_b64_str,
    )


if __name__ == "__main__":
    app.run(debug=True)
