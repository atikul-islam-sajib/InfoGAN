import joblib as pkl
import os
import torch
import torch.nn as nn

def pickle(value=None, filename=None):
    if (value and filename) is not None:
        pkl.dump(value=value, filename=filename)
    else:
        ValueError("Pickle is not possible due to missing arguments".capitalize())


def clean_folder(path=None):
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
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
def device(device = "cpu"):
    if device == "mps":
        return torch.device("mps" if torch.mps.backends.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")
