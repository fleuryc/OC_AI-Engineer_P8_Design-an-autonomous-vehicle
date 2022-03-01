import argparse
import os
import random
import typing
from dataclasses import dataclass
from pathlib import Path

import albumentations as aug
import azureml
import mlflow
import numpy as np
import tensorflow as tf


# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert tf.test.gpu_device_name()


# Enable NVIDIA AMP
# see : https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensorflow
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
# see : https://keras.io/api/mixed_precision/#using-mixed-precision-training-in-keras
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# Enable TensorFlow XLA
# see : https://www.tensorflow.org/versions/r2.7/api_docs/python/tf/config/optimizer/set_jit
os.environ[
    "TF_XLA_FLAGS"
] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices"
tf.config.optimizer.set_jit(True)


@dataclass(frozen=True)
class CityscapesCategory:
    """Cityscapes categories.

    Attributes:
        id: The category id
        name: The category name
        color: The category color
    """

    id: int
    name: str
    color_rgb: typing.Tuple[int, int, int]
    label_ids: typing.List[int]


# Cityscape categories
CITYSCAPES_CATEGORIES = [
    CityscapesCategory(
        0,
        "void",
        (0, 0, 0),  # black
        list(range(0, 7)),
    ),
    CityscapesCategory(
        1,
        "flat",
        (128, 64, 128),  # purple
        list(range(7, 11)),
    ),
    CityscapesCategory(
        2,
        "construction",
        (70, 70, 70),  # dark gray
        list(range(11, 17)),
    ),
    CityscapesCategory(
        3,
        "object",
        (153, 153, 153),  # light gray
        list(range(17, 21)),
    ),
    CityscapesCategory(
        4,
        "nature",
        (107, 142, 35),  # olive
        list(range(21, 23)),
    ),
    CityscapesCategory(
        5,
        "sky",
        (70, 130, 180),  # blue
        list(range(23, 24)),
    ),
    CityscapesCategory(
        6,
        "human",
        (220, 20, 60),  # crimson
        list(range(24, 26)),
    ),
    CityscapesCategory(
        7,
        "vehicle",
        (0, 0, 142),  # navy
        [-1] + list(range(26, 34)),
    ),
]


def cityscapes_label_ids_to_category_ids(img: np.array) -> np.array:
    """Convert cityscapes label ids to category ids.

    Args:
        img: Cityscapes label ids image

    Returns:
        Category ids image
    """
    if not len(img.shape) == 2:
        raise ValueError("Image must be of shape (H, W).")

    category_ids = np.zeros(img.shape, dtype=np.uint8)

    for category in CITYSCAPES_CATEGORIES:
        for label_id in category.label_ids:
            category_ids[img == label_id] = category.id

    return category_ids


def cityscapes_category_ids_to_category_colors(img: np.array) -> np.array:
    """Convert cityscapes category ids to colors.

    Args:
        img: Cityscapes category ids image

    Returns:
        Category colors image
    """
    if not len(img.shape) == 2:
        raise ValueError("Image must be of shape (H, W).")

    category_colors = np.zeros(img.shape + (3,), dtype=np.uint8)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            category_colors[row, col] = CITYSCAPES_CATEGORIES[
                img[row, col]
            ].color_rgb

    return category_colors


class CityscapesGenerator(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(
        self,
        batch_size,
        img_size,
        input_img_paths,
        label_ids_img_paths,
        augment=None,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.label_ids_img_paths = label_ids_img_paths
        self.augment = augment

    def __len__(self):
        return len(self.label_ids_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype=np.uint8)
        for j, path in enumerate(batch_input_img_paths):
            x[j] = np.array(
                tf.keras.utils.load_img(path, target_size=self.img_size)
            )

        batch_label_ids_img_paths = self.label_ids_img_paths[
            i : i + self.batch_size
        ]
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype=np.uint8)
        for j, path in enumerate(batch_label_ids_img_paths):
            y[j] = np.expand_dims(
                cityscapes_label_ids_to_category_ids(
                    np.array(
                        tf.keras.utils.load_img(
                            path,
                            target_size=self.img_size,
                            color_mode="grayscale",
                        )
                    )
                ),
                2,
            )

        if self.augment is not None:
            for j in range(self.batch_size):
                augmented = self.augment(
                    image=x[j],
                    mask=y[j],
                )
                x[j] = augmented["image"]
                y[j] = augmented["mask"]

        return x, y


class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    """Custom metric to report the mean IoU over the entire batch.

    See : https://github.com/tensorflow/tensorflow/issues/32875#issuecomment-707316950
    """

    def __init__(
        self, y_true=None, y_pred=None, num_classes=None, name=None, dtype=None
    ):
        super(UpdatedMeanIoU, self).__init__(
            num_classes=num_classes, name=name, dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def jaccard_loss(y_true, y_pred, smooth=100.0, num_classes=8):
    """
    See : https://towardsdatascience.com/image-segmentation-choosing-the-correct-metric-aa21fd5751af
    """

    y_true = tf.cast(tf.squeeze(tf.one_hot(y_true, num_classes)), tf.float16)
    y_pred = tf.cast(y_pred, tf.float16)

    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)

    return (1 - jac) * smooth


def unet_xception_model(img_size, num_classes, model_name="unet_xception"):
    """Creates a U-Net model with Xception as the encoder.

    See : https://keras.io/examples/vision/oxford_pets_image_segmentation/

    Args:
        img_size: tuple of (height, width)
        num_classes: number of classes to predict

    Returns:
        keras.models.Model
    """
    inputs = tf.keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(
            filters, 1, strides=2, padding="same"
        )(previous_block_activation)
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = tf.keras.layers.Conv2D(
        num_classes, 3, activation="softmax", padding="same"
    )(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


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
        choices=["unet_xception"],
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
                # aug.OneOf(  # Weather augmentations
                #     [
                #         aug.RandomRain(),
                #         aug.RandomFog(),
                #         aug.RandomShadow(),
                #         aug.RandomSnow(),
                #         aug.RandomSunFlare(),
                #     ]
                # ),
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
                "UpdatedMeanIoU": UpdatedMeanIoU,
                "jaccard_loss": jaccard_loss,
            },
        )
        print(f">>> Loaded model {model_name} from Azure ML.")
    except Exception as e:
        print(f">>> Error loading model {model_name} from Azure ML: {e}")
        print(f">>> Creating new model {model_name}.")

        model = None
        if args.model == "unet_xception":
            model = unet_xception_model(
                img_size, num_classes, model_name=model_name
            )

    # Configure the model for training.
    model.compile(
        # Enable NVIDIA AMP
        # see : https://keras.io/api/mixed_precision/loss_scale_optimizer/
        optimizer=tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(learning_rate=1e-2)
        ),
        loss=jaccard_loss,  # "sparse_categorical_crossentropy",
        metrics=[
            UpdatedMeanIoU(
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
        train_label_colors_img_paths = sorted(
            Path(gtFine_path, "train").glob("**/*_gtFine_color.png")
        )

        # Validation dataset
        val_input_img_paths = sorted(
            Path(leftImg8bit_path, "val").glob("**/*_leftImg8bit.png")
        )
        val_label_ids_img_paths = sorted(
            Path(gtFine_path, "val").glob("**/*_gtFine_labelIds.png")
        )
        val_label_colors_img_paths = sorted(
            Path(gtFine_path, "val").glob("**/*_gtFine_color.png")
        )

        # Test dataset
        test_input_img_paths = sorted(
            Path(leftImg8bit_path, "test").glob("**/*_leftImg8bit.png")
        )
        test_label_ids_img_paths = sorted(
            Path(gtFine_path, "test").glob("**/*_gtFine_labelIds.png")
        )
        test_label_colors_img_paths = sorted(
            Path(gtFine_path, "test").glob("**/*_gtFine_color.png")
        )

        with mlflow.start_run() as mlflow_run:
            # train model
            model.fit(
                CityscapesGenerator(
                    batch_size,
                    img_size,
                    train_input_img_paths,
                    train_label_ids_img_paths,
                    augment,
                ),
                validation_data=CityscapesGenerator(
                    batch_size,
                    img_size,
                    val_input_img_paths,
                    val_label_ids_img_paths,
                    augment,
                ),
                epochs=100,
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
                        min_delta=1e-3,
                        verbose=1,
                    ),
                    tf.keras.callbacks.TensorBoard(
                        log_dir=Path(model_path, "./logs")
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
