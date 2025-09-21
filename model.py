from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch
import torchquantum as tq
from torchinfo import summary
import torchquantum.functional as tqf
from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.layers import U3CU3Layer0
import torch.nn as nn
#from torchsummary import summary
import torch.nn.functional as F
from PIL import Image
import time
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
from args import Args


#===========================模型构建模块=============================================#

class TrainableQuanvFilter(tq.QuantumModule):
  def __init__(self):
    super().__init__()
    self.n_wires = 4 # 量子卷子核线路的数量；
    self.encoder = tq.GeneralEncoder(
        [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        ]
    )

    """
    self.encoder = tq.GeneralEncoder(
      [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [4], "func": "rz", "wires": [0]},
        {"input_idx": [5], "func": "rz", "wires": [1]},
        {"input_idx": [6], "func": "rz", "wires": [2]},
        {"input_idx": [8], "func": "rx", "wires": [0]},
        {"input_idx": [9], "func": "rx", "wires": [1]},
        {"input_idx": [10], "func": "rx", "wires": [2]},
        {"input_idx": [12], "func": "ry", "wires": [0]},
        {"input_idx": [13], "func": "ry", "wires": [1]},
        {"input_idx": [14], "func": "ry", "wires": [2]},
      ]
    )
    """

    # "n_blocks" 代表量子卷积线路的深度；
    self.arch = {"n_wires": self.n_wires, "n_blocks": Args.blocks}
    self.q_layer = U3CU3Layer0(self.arch)
    self.measure = tq.MeasureAll(tq.PauliX)


  def forward(self, x, use_qiskit=False):
    bsz = x.shape[0]
    stride = 2
    qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)

    x = x.view(bsz, Args.size, Args.size)

    data_list = []
    for c in range(0, Args.size, stride):
      for r in range(0, Args.size, stride):
        data = torch.transpose(torch.cat(
                (x[:, c, r], x[:, c, r + 1], x[:, c + 1, r], x[:, c + 1, r + 1])
                ).view(4, bsz), 0, 1)
        if use_qiskit:
          data = self.qiskit_processor.process_parameterized(qdev, self.encoder, self.q_layer, self.measure, data)
        else:
          self.encoder(qdev, data)
          self.q_layer(qdev)
          data = self.measure(qdev)
          #print(data.shape) # [16, 4]
        data_list.append(data.view(bsz, 4))
    #print("data_list", len(data_list)) #6 => 9 | 8 => 16

    result = torch.transpose(torch.cat(data_list, dim=-1).view(bsz, Args.size, Args.size), 1, 2).float()

    return result



class Qichir(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=2048, out_channels=512,
              kernel_size=2, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=2))
    self.qf = TrainableQuanvFilter()
    self.fc1 = nn.Sequential(nn.Linear(512*3*3, Args.size*Args.size), nn.ReLU())
    self.fc2 = nn.Linear(Args.size*Args.size, 4)
    self.name = "Qichir"

  def forward(self, x, use_qiskit=False):
    x = x.view(-1, 2048, 7, 7)
    x = self.conv(x)

    x = x.view(-1, 3*3, 512)
    x = x.reshape(x.shape[0], 512*3*3)
    x = self.fc1(x)

    x = x.view(x.shape[0], Args.size, Args.size)
    x = self.qf(x)
    x = x.reshape(x.shape[0], Args.size*Args.size)
    x = self.fc2(x)

    return F.log_softmax(x, -1)




