import matplotlib.pyplot as plt
import numpy as np
import src.data.cityscapes as cityscapes_data
import tensorflow as tf


class CityscapesViewerCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_input_img_paths, val_label_colors_img_paths, img_size):
        self.val_input_img_paths = val_input_img_paths
        self.val_label_colors_img_paths = val_label_colors_img_paths
        self.img_size = img_size

    def on_epoch_end(self, epoch=None, logs=None):
        rand_idx = np.random.randint(0, len(self.val_input_img_paths))

        fig, ax = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18, 6),
        )

        ax[0].title.set_text("Original image")
        val_img = tf.keras.utils.load_img(
            self.val_input_img_paths[rand_idx], target_size=self.img_size
        )
        ax[0].imshow(val_img)

        ax[1].title.set_text("Original label colors")
        val_colors = tf.keras.utils.load_img(
            self.val_label_colors_img_paths[rand_idx], target_size=self.img_size
        )
        ax[1].imshow(val_colors)

        ax[2].title.set_text("Predicted category colors")
        pred_colors = (
            cityscapes_data.cityscapes_category_ids_to_category_colors(
                np.squeeze(
                    np.argmax(
                        self.model.predict(np.expand_dims(val_img, 0)), axis=-1
                    )
                )
            )
        )
        ax[2].imshow(pred_colors)

        plt.show()
