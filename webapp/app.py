from crypt import methods
from flask import Flask, render_template, request

import src.utils as utils

app = Flask(__name__)


@app.route("/", methods=["GET"])
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
            model_base_path="./../results/downlad/",
            model_name="deeplab_v3plus_64",
            images_base_path="./../data/raw",
            image_id=image_id,
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
