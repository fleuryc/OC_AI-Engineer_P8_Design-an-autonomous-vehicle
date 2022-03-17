import argparse
import random
from pathlib import Path

import albumentations as aug
import mlflow
import tensorflow as tf

import azureml
import cityscapes
from models import deeplab_v3plus, unet_xception
from models.keras_segmentation.models import fcn

# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert tf.test.gpu_device_name()


# Enable Automatic Mixed Precision (AMP)
# see : https://keras.io/api/mixed_precision/#using-mixed-precision-training-in-keras
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# Enable Accelerated Linear Algebra (XLA)
# see : https://www.tensorflow.org/versions/r2.7/api_docs/python/tf/config/optimizer/set_jit
tf.config.optimizer.set_jit(True)


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
    num_classes = 8

    augment = None
    if args.augment:
        # Image augmentation : None for no augmentation
        augment = aug.Compose(
            [
                aug.OneOf(  # Color augmentations
                    [
                        aug.RandomBrightnessContrast(),
                        aug.RandomGamma(),
                        aug.RandomToneCurve(),
                    ]
                ),
                aug.OneOf(  # Camera augmentations
                    [
                        aug.MotionBlur(),
                        aug.GaussNoise(),
                    ]
                ),
                aug.OneOf(  # Geometric augmentations
                    [
                        aug.HorizontalFlip(),
                        aug.RandomCrop(
                            width=int(img_size[0] / random.uniform(1.0, 2.0)),
                            height=int(img_size[1] / random.uniform(1.0, 2.0)),
                        ),
                        aug.SafeRotate(
                            limit=15,
                        ),
                    ]
                ),
                aug.Resize(
                    width=img_size[0],
                    height=img_size[1],
                ),
            ]
        )

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
        print(f">>> Creating new model {model_name}.")

        model = None
        if args.model == "unet_xception":
            model = unet_xception.get_model(
                img_size, num_classes, model_name=model_name
            )
        elif args.model == "deeplab_v3plus":
            model = deeplab_v3plus.get_model(
                weights="cityscapes",
                input_tensor=None,
                input_shape=(args.resize, args.resize, 3),
                classes=8,
                backbone="mobilenetv2",
                OS=16,
                alpha=1.0,
                activation="softmax",
                model_name=model_name,
            )
        elif args.model == "fcn_8":
            model = fcn.fcn_8(
                n_classes=8,
                input_height=args.resize,
                input_width=args.resize,
                channels=3,
            )

    # Configure the model for training.
    model.compile(
        optimizer="adam",
        loss=cityscapes.jaccard_loss,  # "sparse_categorical_crossentropy",
        metrics=[
            cityscapes.UpdatedMeanIoU(
                name="MeanIoU", num_classes=num_classes
            ),  # https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html
        ],
    )

    with dataset.mount("./data") as data_mount:
        raw_data_path = Path(data_mount.mount_point)
        leftImg8bit_path = Path(raw_data_path, "leftImg8bit")
        gtFine_path = Path(raw_data_path, "gtFine")

        # Train dataset
        train_input_img_paths = sorted(
            Path(leftImg8bit_path, "train").glob("**/*_leftImg8bit.png")
        )
        train_label_ids_img_paths = sorted(
            Path(gtFine_path, "train").glob("**/*_gtFine_labelIds.png")
        )

        # Validation dataset
        val_input_img_paths = sorted(
            Path(leftImg8bit_path, "val").glob("**/*_leftImg8bit.png")
        )
        val_label_ids_img_paths = sorted(
            Path(gtFine_path, "val").glob("**/*_gtFine_labelIds.png")
        )
        val_label_colors_img_paths = sorted(
            Path(gtFine_path, "val").glob("**/*_color.png")
        )

        with mlflow.start_run() as mlflow_run:
            # train model
            model.fit(
                cityscapes.CityscapesGenerator(
                    batch_size,
                    img_size,
                    train_input_img_paths,
                    train_label_ids_img_paths,
                    augment,
                ),
                validation_data=cityscapes.CityscapesGenerator(
                    batch_size,
                    img_size,
                    val_input_img_paths,
                    val_label_ids_img_paths,
                    augment,
                ),
                epochs=25,
                callbacks=[
                    tf.keras.callbacks.ReduceLROnPlateau(
                        patience=2,
                        factor=0.5,
                        min_delta=1e-2,
                        min_lr=1e-7,
                        verbose=1,
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                        min_delta=1e-2,
                        verbose=1,
                    ),
                    tf.keras.callbacks.TensorBoard(
                        log_dir=Path(model_path, "logs")
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        Path(model_path, "checkpoint.h5"),
                        monitor="val_loss",
                        save_best_only=True,
                        verbose=1,
                    ),
                    cityscapes.CityscapesViewerCallback(
                        val_input_img_paths,
                        val_label_colors_img_paths,
                        img_size,
                    ),
                ],
                # ! Multi-processing makes the training stop at start of epoch 2
                # workers=4,
                # use_multiprocessing=True,
            )

            # register model
            print(f">>> Saving model {model_name} to Azure ML.")
            model_uri = "runs:/{}/model".format(mlflow_run.info.run_id)
            model = mlflow.register_model(model_uri, model_name)


if __name__ == "__main__":
    main()
