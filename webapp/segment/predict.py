from flask import Blueprint, render_template

bp = Blueprint("predict", __name__)


@bp.route("/")
def index():
    return render_template(
        "predict/index.html",
    )
