import pickle
import torch
from my_package.dataset import FeaturesDataset
from my_package import TestRecord, wait_gpu
from my_package import model as Model
from tzy_model import S4Model
from torch.utils.data import DataLoader
from my_package.label import MIX_LABEL_DICT as LABEL_DICT
from my_package.label import NAME_DICT
from tqdm import tqdm
import os

with open("/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/get_wsi_features/models/s4_mix/[phikon2_Medium]s4_mix.pickle")
model = 