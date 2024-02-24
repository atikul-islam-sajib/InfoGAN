import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/cli.log",
)

sys.path.append("src/")

from dataloader import Loader
from discriminator import Discriminator
from generator import Generator
from trainer import Trainer
from test import Test

if __name__ == "__main__":
    """
    This script provides a command line interface (CLI) for training, testing, and managing a generative adversarial network (GAN) model, specifically tailored for MNIST digit data. It facilitates various operations such as data loading, model training, and image generation based on user-defined configurations.

    Parameters:
    - `--batch_size` (int): Defines the batch size for training the model. Default is 128.
    - `--image_size` (int): Specifies the size of images (height and width) to be used. Default is 32.
    - `--in_channels` (int): Number of input channels in the images. This is required for model configuration.
    - `--latent_space` (int): Dimensionality of the latent space from which the generator creates images. Default is 100.
    - `--epochs` (int): Number of training epochs. Default is 10.
    - `--lr` (float): Learning rate for the optimizer. Default is 0.0002.
    - `--display` (bool): Flag to enable or disable display of generated images during training. Default is True.
    - `--device` (str): Specifies the computing device ('cuda' or 'cpu') for model training and inference. Default is 'cuda'.
    - `--num_samples` (int): Number of images to generate for testing. Default is 20.
    - `--data` (action): Flag to indicate the use of MNIST digit data for the model. No default.
    - `--test` (action): Flag to indicate testing mode. No default.

    Methods:
    - Loader: Responsible for loading and preprocessing the dataset.
    - Discriminator: Defines the discriminator component of the GAN.
    - Generator: Defines the generator component of the GAN.
    - Trainer: Manages the training process of the GAN, including both the generator and discriminator.
    - Test: Facilitates testing and evaluation of the trained model by generating images.

    Usage:
    The script is executed from the command line, allowing users to specify configuration options for training and testing the GAN model. Example usage might look like this:

    ```
    python <script_name>.py --batch_size 64 --image_size 32 --in_channels 1 --latent_space 100 --epochs 20 --lr 0.0001 --device cuda --data --test
    ```

    ```
    python <script_name>.py --num_samples 20 --latent_space 100 --device mps --test
    ```

    """
    parser = argparse.ArgumentParser(description="CLI - command line interface".title())
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Define batch size".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=32, help="Define image size".capitalize()
    )
    parser.add_argument(
        "--in_channels", type=int, help="Number of input channels".capitalize()
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Define the latent space".capitalize(),
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Define the epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Define the learning rate".capitalize()
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=True,
        help="Define if you want to display or not".capitalize(),
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Define the device".capitalize()
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of images to generate".capitalize(),
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Mnist digit data for the model".capitalize(),
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.data:
        if (
            args.batch_size
            and args.image_size
            and args.in_channels
            and args.latent_space
            and args.epochs
            and args.lr
            and args.device
            and args.display
        ):
            logging.info("Load the dataset".capitalize())

            loader = Loader(batch_size=args.batch_size, image_size=args.image_size)
            dataloader = loader.download_mnist()

            logging.info("Train the model".capitalize())

            trainer = Trainer(
                epochs=args.epochs,
                in_channels=args.in_channels,
                lr=args.lr,
                latent_space=args.latent_space,
                batch_size=args.batch_size,
                display=args.display,
                device=args.device,
            )
            trainer.train()

            logging.info("Test the model".capitalize())

    if args.test:
        if args.num_samples and args.latent_space and args.device:
            logging.info("Test the model".capitalize())

            test = Test(latent_space=args.latent_space, num_samples=args.num_samples)
            test.test()
