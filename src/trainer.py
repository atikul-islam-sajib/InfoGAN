import sys
import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('src/')

from discriminator import Discriminator
from generator import Generator
from dataloader import loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/trainer.log",
)