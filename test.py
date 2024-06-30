import torch
from utils import save_checkpoint, save_some_examples, load_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
checkpoint_file = config.CHECKPOINT_GEN
if os.path.exists(checkpoint_file) and os.path.getsize(checkpoint_file) > 0:
    load_checkpoint(checkpoint_file, gen, opt_gen, config.LEARNING_RATE)
else:
    print(f"Checkpoint file {checkpoint_file} is missing or empty.")