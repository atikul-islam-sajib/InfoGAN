import joblib as pkl
import os
import torch
import torch.nn as nn


def pickle(value=None, filename=None):
    """
    Serializes and saves a Python object to a file using joblib.

    Parameters:
    - value (any): The Python object to serialize.
    - filename (str): The path to the file where the serialized object will be saved.

    Raises:
    - ValueError: If either `value` or `filename` is not provided.
    """
    if (value and filename) is not None:
        pkl.dump(value=value, filename=filename)
    else:
        ValueError("Pickle is not possible due to missing arguments".capitalize())


def clean_folder(path=None):
    """
    Deletes all files within a specified directory.

    Parameters:
    - path (str): The path to the directory to clean.

    Raises:
    - ValueError: If `path` is not provided.
    """
    if path is not None:
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))

            print("{} - path cleaned".format(path).capitalize())
        else:
            print("{} - path doesn't exist".capitalize())
    else:
        raise ValueError(
            "Clean folder is not possible due to missing arguments".capitalize()
        )


def weight_init(m):
    """
    Initializes the weights of a PyTorch model according to the type of layer.

    Parameters:
    - m (torch.nn.Module): The module (layer) to initialize.

    This function applies a normal distribution initialization to convolutional layers
    and batch normalization layers, with means of 0.0 and 1.0 respectively, and a standard deviation of 0.02.
    """
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def device_init(device="cpu"):
    """
    Initializes and returns a PyTorch device object.

    Parameters:
    - device (str): The desired device to use. Options include "cpu", "cuda", and "mps".

    Returns:
    - torch.device: The PyTorch device object initialized to the specified device or a fallback if the preferred device is not available.
    """
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")
