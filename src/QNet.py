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
    filename="./logs/QNet.log",
)

class QNet(nn.Module):
    """
    A neural network model designed for Q-learning tasks, featuring convolutional layers for feature extraction and fully connected layers for action value prediction.

    The model processes input images through convolutional layers followed by fully connected layers to predict action values for each possible action, suitable for reinforcement learning environments.

    | Parameters   | Description |
    |--------------|-------------|
    | (None)       | This class does not take parameters at initialization. |

    | Attributes   | Description |
    |--------------|-------------|
    | layers_config| list of tuples. Configuration for convolutional layers specifying in_channels, out_channels, kernel_size, stride, padding, and whether to use batch normalization. |
    | model        | nn.Sequential. The sequential model comprising the convolutional layers. |
    | fc_layer     | nn.Sequential. The sequential model comprising the fully connected layers for action value prediction. |

    | Methods      | Description |
    |--------------|-------------|
    | forward(x)   | Defines the forward pass through the convolutional and fully connected layers. |
    | connected_layer(layers_config) | Constructs the convolutional layers based on the provided configuration. |
    | connected_fc_layer(in_features) | Constructs the fully connected layers for action value prediction. |

    Examples
    --------
    >>> qnet = QNet()
    >>> image = torch.randn(64, 1, 32, 32)
    >>> print(qnet(image).shape)
    """
    def __init__(self, num_labels = 10):
        self.num_labels = num_labels
        
        super().__init__()
        
        self.layers_config = [
            (1, 128, 4, 2, 1, True),
            (128, 64, 4, 2, 1, True),
        ]
        
        self.model = self.connected_layer(layers_config = self.layers_config)
        self.fc_layer = self.connected_fc_layer(in_features = 8*8*64)
    
    def connected_layer(self, layers_config = None):
        """
        Constructs the convolutional layers of the model based on the provided configuration.

        Parameters
        ----------
        layers_config : list of tuples, optional
            Configuration for each convolutional layer. If not provided, uses the instance's layers_config.

        Returns
        -------
        nn.Sequential
            A sequential container of the constructed convolutional layers.

        Raises
        ------
        Exception
            If layers_config is not provided.
        """
        layers = OrderedDict()
        if layers_config is not None:
            for index, (in_channels, out_channels, kernel_size, stride, padding, use_norm) in enumerate(layers_config):
                layers[f"conv_{index+1}"] = nn.Conv2d(
                    in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
                if use_norm:
                    layers[f"batchnorm_{index+1}"] = nn.BatchNorm2d(out_channels)
                
                layers[f"relu_{index+1}"] = nn.ReLU(inplace = True)
                
            return nn.Sequential(layers)
        
        else:
            raise Exception("Config layer is not passed".capitalize())
    
    def connected_fc_layer(self, in_features = None):
        """
        Constructs the fully connected layers of the model for action value prediction.

        Parameters
        ----------
        in_features : int, optional
            The number of input features to the first fully connected layer.

        Returns
        -------
        nn.Sequential
            A sequential container of the constructed fully connected layers.

        Raises
        ------
        Exception
            If in_features is not provided.
        """
        layers = OrderedDict()
        layers[f"fc_1"] = nn.Linear(in_features = in_features, out_features = 10)
        layers[f"fc_soft"] = nn.Softmax(dim = 1)
        
        return nn.Sequential(layers)
    
    def forward(self, x):
        """
        Defines the forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the model.

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the convolutional and fully connected layers.

        Raises
        ------
        Exception
            If the input tensor x is not provided.
        """
        if x is not None:
            x = self.model(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layer(x)
            return x
        else:
            raise Exception("Input is not passed".capitalize())
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the QNet".title())
    
    parser.add_argument("--num_labels", type=int, default=10, help="Number of labels".capitalize())
    parser.add_argument("--model", action="store_true", help="Define the model".capitalize())
    
    args = parser.parse_args()
    
    if args.model:
        logging.info("Defining the QNet".capitalize())
        if args.num_labels:
            qnet = QNet()
            
            logging.info("Testing the QNet".capitalize())
            
            image = torch.randn(64, 1, 32, 32)
            logging.info(qnet(image).shape)
        else:
            raise Exception("Number of labels is not provided".capitalize())
    else:
        raise Exception("Model is not defined".capitalize())