from model import *
from utils import *
from args import *
from feature_extraction import feature_extract
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
from torch.utils.data.dataset import Dataset


class LpDataset(Dataset):

  def __init__(self, imgs, labels):

    self.labels = labels
    self.imgs = imgs

  def __getitem__(self, idx):
    img, label = self.imgs[idx], self.labels[idx]

    return img, label

  def __len__(self):
    return len(self.labels)


def train(trainLoader, model, device, optimizer):
  for idx, trainD in enumerate(trainLoader):
    inputs, targets = trainD
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = F.nll_loss(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info("loss: %f"%loss.item())
    #print(f"loss: {loss.item()}", end="\r")


def valid(validLoader, model, device, qiskit=False):
  target_all = []
  output_all = []
  with torch.no_grad():
    for idx, validD in enumerate(validLoader):
      inputs, targets = validD
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)

      target_all.append(targets)
      output_all.append(outputs)
    target_all = torch.cat(target_all, dim=0)
    output_all = torch.cat(output_all, dim=0)
    #print(output_all)

  _, indices = output_all.topk(1, dim=1)
  masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
  size = target_all.shape[0]
  corrects = masks.sum().item()
  accuracy = corrects / size
  loss = F.nll_loss(output_all, target_all).item()

  logger.info("|| acc of valid set: %f"%accuracy)
  logger.info("|| loss of valid set: %f"%loss)

  return accuracy, loss, target_all, output_all


def test(testLoader, model, device, qiskit=False):
  target_all = []
  output_all = []
  with torch.no_grad():
    for idx, testD in enumerate(testLoader):
      inputs, targets = testD
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)

      target_all.append(targets)
      output_all.append(outputs)
    target_all = torch.cat(target_all, dim=0)
    output_all = torch.cat(output_all, dim=0)

  _, indices = output_all.topk(1, dim=1)
  masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
  size = target_all.shape[0]
  corrects = masks.sum().item()
  accuracy = corrects / size
  loss = F.nll_loss(output_all, target_all).item()

  print("|| acc of test set: %f"%accuracy)
  print("|| loss of test set: %f"%loss)

  return accuracy, loss, target_all, output_all


def get_logger(name):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)

  handler_stdout = logging.StreamHandler()
  handler_stdout.setLevel(logging.INFO)
  handler_stdout.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
  logger.addHandler(handler_stdout)

  logger.removeHandler(handler_stdout)

  return logger


if __name__ == '__main__':

    np.random.seed(42)
    torch.manual_seed(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    times3 = []
    report_lis = []
    best_dics = []
    val_acc = 0.0

    logger = get_logger(__name__)

    # ich_name: "nh", "ns", "xiu"
    trainX, trainY, testX, testY, validX, validY = feature_extract("ich_name")
    trainData = LpDataset(imgs=trainX, labels=trainY)
    validData = LpDataset(imgs=validX, labels=validY)
    testData = LpDataset(imgs=testX, labels=testY)

    trainLoader = DataLoader(dataset=trainData, batch_size=Args.bsz,
                             sampler=torch.utils.data.RandomSampler(trainData),
                             num_workers=8,
                             pin_memory=True)
    validLoader = DataLoader(dataset=validData, batch_size=Args.bsz,
                             sampler=torch.utils.data.RandomSampler(validData),
                             num_workers=8,
                             pin_memory=True)
    testLoader = DataLoader(dataset=testData, batch_size=Args.bsz,
                            sampler=torch.utils.data.RandomSampler(testData),
                            num_workers=8,
                            pin_memory=True)

    print("Args.size", Args.size, "\n", "Args.blocks", Args.blocks, "\n", "Args.bsz", Args.bsz)
    for t in range(1, 4):
        model = Qichir().to(device)
        print(">>>>>>>>>>>>The %sth time<<<<<<<<<<<<" % t)
        print("|| Train %s ||" % model.name)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=Args.epochs)
        accs, y_preds, y_trues = [], [], []
        times = []
        report_dic_lis = []

        for epoch in range(1, Args.epochs + 1):
            # train
            print("******Train %sth epoch*******" % epoch)
            s_time = time.time()
            train(trainLoader, model, device, optimizer)
            e_time = time.time()
            print("|| Training time: %.1f" % (e_time - s_time))
            times.append((e_time - s_time))

            # valid
            acc_val, loss_val, y_true_val, y_pred_val = valid(validLoader, model, device)
            try:
                y_true_val_, y_pred_val_ = y_true_val.cpu(), y_pred_val.cpu().argmax(axis=-1)
            except:
                y_true_val_, y_pred_val_ = y_true_val, y_pred_val.argmax(axis=-1)
            print(classification_report(y_true_val_, y_pred_val_, digits=4))

            report_dic = classification_report(y_true_val_, y_pred_val_, digits=4, output_dict=True)
            report_dic_lis.append(report_dic)

            if report_dic["accuracy"] >= val_acc:
                model_path = "path/to/weights/" + model.name + "_weights.pt"
                val_acc = report_dic["accuracy"]
                torch.save(model, model_path)
                best_dic = report_dic
            else:
                continue

            # test
            acc, loss, y_true, y_pred = test(testLoader, model, device)

            # acc = accuracy_score(y_true, y_pred.argmax(axis=-1))
            y_trues.append(y_true)
            y_preds.append(y_pred)
            accs.append(acc)
            scheduler.step()

        print(best_dic)
        best_dics.append(best_dic)
        report_lis.append(report_dic_lis)
        times3.append([y_trues, y_preds])
        print("|| Average training time: %.1f" % (sum(times) / len(times)))
