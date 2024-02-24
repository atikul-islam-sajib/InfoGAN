import sys
import logging
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path.append("src/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/discriminator.log",
)


class Discriminator(nn.Module):
    """
    A discriminator model for a Generative Adversarial Network (GAN) that classifies images as real or fake.

    This model applies a series of convolutional layers to an input image and outputs a single scalar indicating the likelihood of the image being real.

    | Parameters   | Description |
    |--------------|-------------|
    | in_channels  | int, default=1. The number of channels in the input images. |

    | Attributes   | Description |
    |--------------|-------------|
    | in_channels  | int. The number of channels in the input images. |
    | config_layer | list of tuples. Configuration for each layer in the model, specifying layer parameters. |
    | model        | nn.Sequential. The sequential model comprising the discriminator's layers. |

    | Methods      | Description |
    |--------------|-------------|
    | forward(x)   | Defines the forward pass of the discriminator. |
    | connected_layer(config_layer) | Constructs the layers of the discriminator based on the provided configuration. |

    Examples
    --------
    >>> discriminator = Discriminator(in_channels=1)
    >>> image = torch.randn(64, 1, 28, 28)
    >>> print(discriminator(image).shape)
    """

    def __init__(self, in_channels=1):
        self.in_channels = in_channels
        super(Discriminator, self).__init__()

        self.config_layer = [
            (1, 128, 4, 2, 1, 0.2, False),
            (128, 256, 4, 2, 1, 0.2, True),
            (256, 512, 4, 2, 1, 0.2, True),
            (512, 1, 4, 1, 0),
        ]

        self.model = self.connected_layer(config_layer=self.config_layer)

    def connected_layer(self, config_layer=None):
        """
        Constructs the layers of the discriminator based on the provided configuration.

        Parameters
        ----------
        config_layer : list of tuples, optional
            The configuration for each layer in the model. If not provided, uses the instance's config_layer.

        Returns
        -------
        nn.Sequential
            A sequential container of the constructed layers.

        Raises
        ------
        Exception
            If config_layer is not provided.
        """
        layers = OrderedDict()
        if config_layer is not None:
            for index, (
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                slope,
                batch_norm,
            ) in enumerate(config_layer[:-1]):
                layers[f"conv_{index+1}"] = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                if batch_norm:
                    layers[f"batch_norm_{index+1}"] = nn.BatchNorm2d(out_channels)

                layers[f"leaky_relu_{index+1}"] = nn.LeakyReLU(slope)

            (in_channels, out_channels, kernel_size, stride, padding) = config_layer[-1]
            layers["out_conv"] = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
            layers["out_sigmoid"] = nn.Sigmoid()

            return nn.Sequential(layers)
        else:
            raise Exception("Config layer is not passed".capitalize())

    def forward(self, x):
        """
        Defines the forward pass of the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the discriminator.

        Returns
        -------
        torch.Tensor
            The output tensor after processing the input through the discriminator model.

        Raises
        ------
        Exception
            If the input tensor x is not provided.
        """
        if x is not None:
            x = self.model(x)
            return x.view(x.size(0), -1)
        else:
            raise Exception("Input is not passed".capitalize())

    @classmethod
    def total_params(cls):
        return sum(p.numel() for p in cls.parameters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator Configuration".title())
    parser.add_argument(
        "--in_channels", type=int, help="Number of input channels".capitalize()
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="Define the discriminator model".capitalize(),
    )

    args = parser.parse_args()

    if args.model:
        logging.info("Defining the discriminator model".capitalize())
        if args.in_channels is not None:
            discriminator = Discriminator(in_channels=args.in_channels)

            logging.info("Discriminator model defined successfully".capitalize())
            image = torch.randn(64, 1, 28, 28)
            logging.info(discriminator(image).shape)

            logging.info(Discriminator.total_params())
        else:
            raise Exception("Number of input channels is not provided".capitalize())
    else:
        raise Exception("Discriminator model is not defined".capitalize())
