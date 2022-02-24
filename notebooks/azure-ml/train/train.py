"""
@See : https://github.com/Azure/azureml-examples/blob/main/python-sdk/workflows/train/tensorflow/mnist-distributed/src/train.py
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

import mlflow
import numpy as np
import tensorflow as tf
from azureml.core import Dataset, Run, Model

from tensorflow.keras import Input, Model, layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import Sequence


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
    color_rgb: Tuple[int, int, int]
    label_ids: List[int]


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


class CityscapesGenerator(Sequence):
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
            x[j] = np.array(load_img(path, target_size=self.img_size))

        batch_label_ids_img_paths = self.label_ids_img_paths[
            i : i + self.batch_size
        ]
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype=np.uint8)
        for j, path in enumerate(batch_label_ids_img_paths):
            y[j] = np.expand_dims(
                cityscapes_label_ids_to_category_ids(
                    np.array(
                        load_img(
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


class UpdatedMeanIoU(MeanIoU):
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

    y_true = tf.squeeze(tf.one_hot(y_true, num_classes))

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
    inputs = Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
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
        "--resize",
        type=int,
        choices=[64, 80, 128, 160, 256, 320, 512, 640, 800, 1024],
        default=160,
        help="Size in pixels of the square input images will be resized to.",
    )
    parser.add_argument(
        "--augment",
        type=bool,
        choices=[False, True],
        default=False,
        help="Whether or not apply image augmentation during model training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet_xception",
        choices=["unet_xception"],
        help="Name of the model.",
    )

    args = parser.parse_args()

    img_size = (args.resize, args.resize)
    num_classes = 8
    batch_size = 32

    # connect to your workspace
    exp_run = Run.get_context()
    ws = exp_run.experiment.workspace

    # set up MLflow to track the metrics
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.set_experiment(args.experiment)
    mlflow.autolog()

    # Get a named dataset from the current workspace
    dataset_name = "cityscapes"
    dataset = Dataset.get_by_name(ws, name=dataset_name)

    # Get the model
    model_name = (
        f"unet_xception_{args.resize}{'_augmented' if args.augment  else ''}"
    )
    model_path = Path("outputs/", model_name)

    try:
        aml_model = Model(ws, model_name)
        aml_model.download(target_dir=Path(model_path, 'download'))
        model = tf.keras.models.load_model(Path(model_path, 'download', "model/data/model"))
    except:
        model = unet_xception_model(img_size, num_classes, model_name=model_name)

    # Configure the model for training.
    model.compile(
        optimizer="adam",
        loss=jaccard_loss,  # "sparse_categorical_crossentropy",
        metrics=[
            UpdatedMeanIoU(
                name="MeanIoU", num_classes=num_classes
            ),  # https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html
        ],
        run_eagerly=True,
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

        with mlflow.start_run(nested=True) as mlflow_run:

            # train model
            model.fit(
                CityscapesGenerator(
                    batch_size,
                    img_size,
                    train_input_img_paths,
                    train_label_ids_img_paths,
                ),
                validation_data=CityscapesGenerator(
                    batch_size,
                    img_size,
                    val_input_img_paths,
                    val_label_ids_img_paths,
                ),
                epochs=100,
                callbacks=[
                    ReduceLROnPlateau(
                        patience=1,
                        factor=0.5,
                        min_delta=1e-3,
                        min_lr=1e-9,
                        verbose=1,
                    ),
                    EarlyStopping(
                        patience=3,
                        restore_best_weights=True,
                        min_delta=1e-5,
                        verbose=1,
                    ),
                    TensorBoard(log_dir=Path(model_path, "logs")),
                ],
                # workers=4,
                # use_multiprocessing=True,
            )

            # register model
            model_uri = "runs:/{}/model".format(mlflow_run.info.run_id)
            model = mlflow.register_model(model_uri, model_name)


if __name__ == "__main__":
    main()
