import sys
import os
import logging
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/generator.log",
)


class Generator(nn.Module):
    """
    A generator model for a Generative Adversarial Network (GAN) that generates images from a latent space input.

    This model constructs images by applying a series of transposed convolutional layers to a latent space vector.

    | Parameters    | Description |
    |---------------|-------------|
    | latent_space  | int, default=100. The dimensionality of the latent space input. |

    | Attributes    | Description |
    |---------------|-------------|
    | latent_space  | int. The dimensionality of the latent space input. |
    | layers_config | list of tuples. Configuration for each layer in the model, specifying layer parameters. |
    | model         | nn.Sequential. The sequential model comprising the generator's layers. |

    | Methods       | Description |
    |---------------|-------------|
    | forward(x)    | Defines the forward pass of the generator. |
    | connected_layer(layers_config) | Constructs the layers of the generator based on the provided configuration. |

    Examples
    --------
    >>> generator = Generator(latent_space=100)
    >>> noise_data = torch.randn(64, 100, 1, 1)
    >>> print(generator(noise_data).shape)
    """

    def __init__(self, latent_space=100):
        """
        Initializes the Generator model with a specified latent space dimensionality.

        Parameters
        ----------
        latent_space : int, optional
            The dimensionality of the latent space input. Default is 100.
        """
        self.latent_space = latent_space
        super(Generator, self).__init__()

        self.layers_config = [
            (self.latent_space, 512, 4, 1, 0, True),
            (512, 256, 4, 2, 1, True),
            (256, 128, 4, 2, 1, True),
            (128, 1, 4, 2, 1),
        ]

        self.model = self.connected_layer(layers_config=self.layers_config)

    def connected_layer(self, layers_config=None):
        """
        Constructs the layers of the generator based on the provided configuration.

        Parameters
        ----------
        layers_config : list of tuples, optional
            The configuration for each layer in the model. If not provided, uses the instance's layers_config.

        Returns
        -------
        nn.Sequential
            A sequential container of the constructed layers.

        Raises
        ------
        Exception
            If layers_config is not provided.
        """
        layers = OrderedDict()
        if layers_config is not None:
            for index, (
                in_channels,out_channels,kernel_size,stride,padding,batch_norm) in enumerate(layers_config[:-1]):
                layers["conv_transpose_{}".format(index + 1)] = nn.ConvTranspose2d(
                    in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                )
                if batch_norm:
                    layers["batch_norm_{}".format(index + 1)] = nn.BatchNorm2d(
                        num_features=out_channels
                    )

                layers[f"relu_{index+1}"] = nn.ReLU(inplace=True)

            (in_channels, out_channels, kernel_size, stride, padding) = layers_config[
                -1
            ]
            layers[f"out_conv"] = nn.ConvTranspose2d(
                in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
            )
            layers[f"out_tanh"] = nn.Tanh()

            return nn.Sequential(layers)

        else:
            raise Exception("Config layer is not passed".capitalize())

    def forward(self, x):
        """
        Defines the forward pass of the generator.

        Parameters
        ----------
        x : torch.Tensor
            The latent space input tensor for the generator.

        Returns
        -------
        torch.Tensor
            The output tensor after processing the input through the generator model.

        Raises
        ------
        Exception
            If the input tensor x is not provided.
        """
        if x is not None:
            x = self.model(x)
            return x
        else:
            raise Exception("Input is not passed".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator configuration".title())
    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Define the latent space".capitalize(),
    )
    parser.add_argument(
        "--model", action="store_true", help="Define the model".capitalize()
    )
    args = parser.parse_args()

    if args.model:
        logging.info("Generator model is being created".capitalize())
        if args.latent_space:
            generator = Generator(latent_space=args.latent_space)

            logging.info("Generator model is created".capitalize())

            noise_data = torch.randn(64, 100, 1, 1)
            logging.info(generator(noise_data).shape)
        else:
            raise Exception("Latent space is not passed".capitalize())
    else:
        raise Exception("Model is not passed".capitalize())
