#%%
import os
import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
import requests
import math
from collections import defaultdict
from model import *
import data_loader
# %%
def acc_table(file_name,name=None):
    if not name:
        name = file_name.split("/")[1].split("_")[-1]
    with open(file_name,"r") as f:
        all_lines = f.readlines()
        all_lines = [l.strip("\n") for l in all_lines]
    title =  "| Type | A-W | D-W | W-D | A-D | D-A | W-A | 31-Avg | A-C | A-P | A-R | C-A | C-P | C-R | P-A | P-C | P-R | R-A | R-C | R-P | Home-Avg |"
    all_spliter = "| - | - | - | - | - | - |- | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"
    base_line = "| Base | 93.6% | 98.3% | 100.0% | 90.2% | 73.5% | 74.8% | 88.4% | 54.4% | 70.8% | 75.4% | 60.4% | 67.8% | 68.0% | 62.6% | 55.9% | 78.5% | 73.8% | 60.6% | 83.1% | 67.6% |"
    all_types = [l.strip(" ") for l in title.split("|")][2:-1]
    infos = defaultdict(float)
    for line in all_lines:
        info = line.split(":")
        infos[info[0]] = float(info[1][:-1])
    office_31 = [infos[key] for key in all_types[:6]]
    office_home = [infos[key] for key in all_types[7:-1]]
    avg_31 = sum(office_31)/len(office_31)
    avg_home = sum(office_home)/len(office_home)
    cur_line = '| '+' | '.join([name] + ['{:.1f}%'.format(acc) for acc in office_31 + [avg_31] + office_home + [avg_home]]) + ' |'
    return "\n".join([title,all_spliter,base_line,cur_line])
print(acc_table("output/2023-08-23_train_target/2023-08-23_final.log"))
# %%
