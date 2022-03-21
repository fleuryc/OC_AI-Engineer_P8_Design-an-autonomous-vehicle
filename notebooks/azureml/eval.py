import os
import argparse
from pathlib import Path

import mlflow
import tensorflow as tf

import azureml
import cityscapes

# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert tf.test.gpu_device_name()


# Enable Automatic Mixed Precision (AMP)
# see : https://keras.io/api/mixed_precision/#using-mixed-precision-training-in-keras
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# Enable Accelerated Linear Algebra (XLA)
# see : https://www.tensorflow.org/versions/r2.7/api_docs/python/tf/config/optimizer/set_jit
tf.config.optimizer.set_jit(True)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="oc-p8-experiment-0",
        help="Name of the Azure ML experiment.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet_xception",
        choices=["unet_xception", "deeplab_v3plus", "fcn_8"],
        help="Name of the model.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        choices=[64, 80, 128, 160, 256, 320, 512, 640, 800, 1024],
        default=160,
        help="Size in pixels of the square input images will be resized to.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=[8, 16, 32, 64, 128],
        default=32,
        help="Batch size to use during training.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply image augmentation during model training.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_false",
        help="No image augmentation during model training.",
    )
    parser.set_defaults(augment=False)
    args = parser.parse_args()

    print(">>> Arguments:")
    for k, v in vars(args).items():
        print(">>>   {}: {}".format(k, v))

    img_size = (args.resize, args.resize)
    batch_size = args.batch

    # connect to your workspace
    exp_run = azureml.core.Run.get_context()
    ws = exp_run.experiment.workspace

    # set up MLflow to track the metrics
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.set_experiment(args.experiment)
    mlflow.autolog()

    # Get a named dataset from the current workspace
    dataset_name = "cityscapes"
    dataset = azureml.core.Dataset.get_by_name(ws, name=dataset_name)

    # Get the model
    model_name = (
        f"{args.model}_{args.resize}{'_augment' if args.augment else ''}"
    )
    model_path = Path("outputs/", model_name)

    try:
        aml_model = azureml.core.model.Model(ws, model_name)
        aml_model.download(
            target_dir=Path(model_path, "download"), exist_ok=True
        )
        model = tf.keras.models.load_model(
            Path(model_path, "download", "model/data/model"),
            custom_objects={
                "UpdatedMeanIoU": cityscapes.UpdatedMeanIoU,
                "jaccard_loss": cityscapes.jaccard_loss,
            },
        )
        print(f">>> Loaded model {model_name} from Azure ML.")
    except Exception as e:
        print(f">>> Error loading model {model_name} from Azure ML: {e}")

    with dataset.mount("./data") as data_mount:
        raw_data_path = Path(data_mount.mount_point)
        leftImg8bit_path = Path(raw_data_path, "leftImg8bit")
        gtFine_path = Path(raw_data_path, "gtFine")

        # Validation dataset
        val_input_img_paths = sorted(
            Path(leftImg8bit_path, "val").glob("**/*_leftImg8bit.png")
        )
        val_label_ids_img_paths = sorted(
            Path(gtFine_path, "val").glob("**/*_gtFine_labelIds.png")
        )

        # eval model
        model.evaluate(
            cityscapes.CityscapesGenerator(
                batch_size,
                img_size,
                val_input_img_paths,
                val_label_ids_img_paths,
            ),
            verbose=1,
            # ! Multi-processing makes the training stop at start of epoch 2
            workers=4,
            use_multiprocessing=True,
        )


if __name__ == "__main__":
    main()
