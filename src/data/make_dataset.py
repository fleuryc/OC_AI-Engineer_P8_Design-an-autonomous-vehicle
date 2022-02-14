"""
Make Dataset CLI.
This file defines the `make_dataset` command line.

usage: make_dataset.py [-h] [-t TARGET_PATH] [-d DATASET_NAME]

Download Cityscapes data and save it to the target directory.

optional arguments:
  -h, --help            show this help message and exit
  -t TARGET_PATH, --target_path TARGET_PATH
                        path to the directory where the data should be saved
  -d DATASET_NAME, --dataset_name DATASET_NAME
                        name of Azure dataset to register
"""
import argparse
import logging
import os
from pathlib import Path

from azureml.core import Dataset, Workspace
from azureml.data.datapath import DataPath
from dotenv import load_dotenv

# Import custom helper libraries
import src.data.helpers as data_helpers


def main():
    if not Path(LOCAL_DATA_PATH).exists:
        logging.info("Creating %s", LOCAL_DATA_PATH)
        Path.makedirs(LOCAL_DATA_PATH)

    for dataset_name, dataset_zip in DATASET_ZIPS.items():
        logging.info("Download and extract dataset : %s", dataset_name)
        data_helpers.download_extract_zip(
            zip_file_url=dataset_zip["url"],
            files_names=dataset_zip["files"],
            target_path=LOCAL_DATA_PATH,
        )

    # Create or load Workspace object
    logging.info("Loading AzureML Workspace : %s", AZURE_WORKSPACE_NAME)
    ws = Workspace(
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    # Fetch the default Datastore
    logging.info("Fetching AzureML default Datastore")
    blob_datastore = ws.get_default_datastore()

    # Create Dataset object
    logging.info("Uploading files to AzureML Dataset : %s", AZURE_DATASET_NAME)
    cityscapes_dataset = Dataset.File.upload_directory(
        src_dir=LOCAL_DATA_PATH,
        target=DataPath(blob_datastore, AZURE_DATASET_NAME),
        overwrite=False,
        show_progress=True,
    )

    # Register the Dataset in the Workspace
    logging.info("Registering AzureML Dataset : %s", AZURE_DATASET_NAME)
    cityscapes_dataset = cityscapes_dataset.register(
        workspace=ws, name=AZURE_DATASET_NAME
    )


if __name__ == "__main__":
    # Read the command line arguments
    parser = argparse.ArgumentParser(
        description="Download data and save it to the target directory \
            and Azure Dataset."
    )
    parser.add_argument(
        "-t",
        "--target_path",
        type=str,
        help="path to the directory where the data should be saved",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        help="name of Azure dataset to register",
    )
    args = parser.parse_args()

    # Set the global variables
    DATASET_ZIPS = {
        "gtFine": {
            "url": "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_gtFine_trainvaltest.zip",  # noqa E501
            "files": [
                "gtFine/val/munster/munster_000173_000019_gtFine_polygons.json",
                "gtFine/train/hanover/hanover_000000_034015_gtFine_labelIds.png",
                "gtFine/test/berlin/berlin_000000_000019_gtFine_color.png",
            ],
        },
        "leftImg8bit": {
            "url": "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_leftImg8bit_trainvaltest.zip",  # noqa E501
            "files": [
                "leftImg8bit/val/munster/munster_000173_000019_leftImg8bit.png",
                "leftImg8bit/train/hanover/hanover_000000_034015_leftImg8bit.png",
                "leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png",
            ],
        },
    }
    LOCAL_DATA_PATH = args.target_path
    AZURE_DATASET_NAME = args.dataset_name

    # Load environment variables from .env file
    load_dotenv()
    AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
    AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
    AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main()
