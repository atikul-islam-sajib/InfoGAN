import os
import sys
import logging
import argparse
import joblib as pkl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/dataloader.log",
)

sys.path.append("src/")

from utils import pickle, clean_folder
from config import RAW_PATH, PROCESSED_PATH


class Loader:
    """
    A class for loading and preprocessing the MNIST dataset.

    This class handles the downloading of the MNIST dataset, performs image transformations, and organizes the data into batches for training or testing.

    | Parameters | Description |
    |------------|-------------|
    | batch_size | int, default=128. The number of samples to include in each batch of data. |

    | Attributes | Description |
    |------------|-------------|
    | batch_size | int. The size of the batch of data. |

    | Methods    | Description |
    |------------|-------------|
    |_do_transformation() | Applies a series of transformations to the dataset images. |
    | download_mnist()    | Downloads the MNIST dataset, applies transformations, and organizes the data into batches. |

    Examples
    --------
    >>> loader = Loader(batch_size=128)
    >>> dataloader = loader.download_mnist()
    """

    def __init__(self, batch_size=128, image_size=32):
        """
        Initializes the Loader with a specified batch size.

        Parameters
        ----------
        batch_size : int, optional
            The number of samples per batch. Default is 128.
        """
        self.batch_size = batch_size
        self.image_size = image_size

    def _do_transformation(self):
        """
        Apply transformations to the dataset images.

        Returns
        -------
        torchvision.transforms.Compose
            A composed series of transformations for image processing.
        """
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        return transform

    def download_mnist(self):
        """
        Download the MNIST dataset and prepare it for training.

        Checks for dataset existence, downloads if necessary, applies transformations, and prepares a DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            A DataLoader containing the preprocessed MNIST dataset in batches.

        Raises
        ------
        Exception
            If any errors occur during the folder cleaning or dataset processing steps.
        """
        if os.path.exists(RAW_PATH):
            try:
                clean_folder(path=RAW_PATH)
            except Exception as e:
                print("Exception caught in the section # {}".format(e))
            else:
                dataloader = datasets.MNIST(
                    root=os.path.join(RAW_PATH),
                    train=True,
                    download=True,
                    transform=self._do_transformation(),
                )
                dataloader = DataLoader(
                    dataset=dataloader, batch_size=self.batch_size, shuffle=True
                )

                try:
                    if os.path.exists(PROCESSED_PATH):
                        try:
                            clean_folder(path=PROCESSED_PATH)
                        except Exception as e:
                            print("Exception caught in the section # {}".format(e))
                        else:
                            pickle(
                                value=dataloader,
                                filename=os.path.join(PROCESSED_PATH, "dataloader.pkl"),
                            )
                    else:
                        os.makedirs(PROCESSED_PATH)
                        print(
                            "Processed path is created in the data folder & run the code again".capitalize()
                        )

                except Exception as e:
                    print("Exception caught in the section # {}".format(e))
                else:
                    return dataloader
        else:
            os.makedirs(RAW_PATH)
            print(
                "raw folder is created in the data folder and run again this code".capitalize()
            )

    @staticmethod
    def quantity_data():
        """
        Calculate the total quantity of data points in the processed dataset.

        Returns
        -------
        int
            The total number of data points across all batches in the DataLoader.

        Notes
        -----
        This method requires the processed dataset to be saved as a pickle file at the location specified by `PROCESSED_PATH`.
        """
        dataloader = pkl.load(filename=os.path.join(PROCESSED_PATH, "dataloader.pkl"))
        return sum(data.shape[0] for data, _ in dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating the dataloader".title())
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Define batch size".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=32, help="Define image size".capitalize()
    )
    parser.add_argument(
        "--data", action="store_true", help="Download the data".capitalize()
    )

    args = parser.parse_args()

    if args.data:
        logging.info("Downloading the data".capitalize())
        if (args.batch_size and args.image_size) is not None:
            loader = Loader(batch_size=args.batch_size)
            dataloader = loader.download_mnist()
            logging.info("Data downloaded successfully".capitalize())

            logging.info("Quantity of the dataset # {}".format(Loader.quantity_data()))
        else:
            raise ValueError("Please provide the batch size".capitalize())
    else:
        raise ValueError("Please provide the data".capitalize())
