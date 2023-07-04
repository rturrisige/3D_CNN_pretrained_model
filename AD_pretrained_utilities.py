import torch
from torchvision import transforms
import torch.nn as nn
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.utils.data import Dataset
tonpy = lambda x: x.detach().cpu().numpy()
from alive_progress import alive_bar

from scipy import ndimage
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statistics import stdev
from numpy import average as avg
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

orig_cmap = plt.cm.Blues
colors = orig_cmap(np.linspace(0.25, 1))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colors)


######################
#  CONFIGURATION    ##
######################

class CNN(nn.Module):
    def __init__(self, param):
        super(CNN, self).__init__()
        self.embedding = nn.ModuleList()
        for i in range(param.n_conv):
            pad = tuple([int((k-1)/2) for k in param.kernels[i]])
            if param.pooling[i] != (0, 0, 0):
                self.embedding.append(nn.Sequential(
                        nn.Conv3d(in_channels=param.in_channels[i], out_channels=param.out_channels[i],
                                  kernel_size=param.kernels[i], stride=(1, 1, 1), padding=pad, bias=False),
                        nn.BatchNorm3d(param.out_channels[i]),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(param.pooling[i], stride=param.pooling[i])))
            else:
                self.embedding.append(nn.Sequential(
                    nn.Conv3d(in_channels=param.in_channels[i], out_channels=param.out_channels[i],
                              kernel_size=param.kernels[i], stride=(1, 1, 1), padding=pad, bias=False),
                    nn.BatchNorm3d(param.out_channels[i]),
                    nn.ReLU(inplace=True)))
        self.ReLU = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=param.dropout)
        self.f = nn.ModuleList()
        for i in range(len(param.fweights)-1):
            self.f.append(nn.Linear(param.fweights[i], param.fweights[i+1]))
    def forward(self, x, return_conv=False):
        out = self.embedding[0](x)
        if return_conv:
            all_layers = [out]
        for i in range(1, len(self.embedding)):
            out = self.embedding[i](out)
            if return_conv:
                all_layers.append(out)
        out = out.view(out.size(0), -1)
        # out = self.Dropout(out)
        for fc in self.f[:-1]:
            out = fc(out)
            out = self.ReLU(out)
            out = self.Dropout(out)
        out = self.f[-1](out)
        if return_conv:
            return F.softmax(out, dim=1), all_layers
        else:
            return F.softmax(out, dim=1)



def compute_output_size(i, K, P, S):
    output_size = ((i - K + 2*P)/S) + 1
    return int(output_size)


class CNN_8CL_B(object):
    def __init__(self):
        self.input_dim = [73, 96, 96]
        self.out_channels = [8, 8, 16, 16, 32, 32, 64, 64]
        self.in_channels = [1] + [nch for nch in self.out_channels[:-1]]
        self.n_conv = len(self.out_channels)
        self.kernels = [(3, 3, 3)] * self.n_conv
        self.pooling = [(4, 4, 4), (0, 0, 0), (3, 3, 3), (0, 0, 0), (2, 2, 2),
                            (0, 0, 0), (2, 2, 2), (0, 0, 0)]
        for i in range(self.n_conv):
            for d in range(3):
                if self.pooling[i][d] != 0:
                    self.input_dim[d] = compute_output_size(self.input_dim[d], self.pooling[i][d], 0, self.pooling[i][d])
        out = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]
        self.fweights = [self.out_channels[-1] * out, 2]
        self.dropout = 0.0

##
# ######################
#     DATA LOADING    ##
# ######################

def resize_data_volume_by_scale(data, scale):
    """
    Resize the data based on the provided scale
    :param scale: float between 0 and 1
    """
    if isinstance(scale, float):
        scale_list = [scale, scale, scale]
    else:
        scale_list = scale
    return ndimage.interpolation.zoom(data, scale_list, order=0)

def normalize_intensity(img_tensor, normalization="mean"):
   """
   Accepts an image tensor and normalizes it
   :param normalization: choices = "max", "mean" , type=str
   For mean normalization we use the non zero voxels only.
   """
   if normalization == "mean":
       mask = img_tensor.ne(0.0)
       desired = img_tensor[mask]
       mean_val, std_val = desired.mean(), desired.std()
       img_tensor = (img_tensor - mean_val) / std_val
   elif normalization == "max":
     MAX, MIN = img_tensor.max(), img_tensor.min()
     img_tensor = (img_tensor - MIN) / (MAX - MIN)
   return img_tensor


def torch_norm(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_tensor = normalize_intensity(input_tensor)
    return input_tensor.unsqueeze_(0)


def img_processing(image, scaling=0.5, final_size=[96, 96, 73]):
    image = resize_data_volume_by_scale(image, scale=scaling)
    new_scaling = [final_size[i] / image.shape[i] for i in range(3)]
    final_image = resize_data_volume_by_scale(image, scale=new_scaling)
    return final_image


class loader(Dataset):
    def __init__(self, dataset, nclasses=2, transform=None, preprocessing=False):
        """
        dataset : list of all filenames
        transform (callable) : a function/transform that acts on the images (e.g., normalization).
        """
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.transform = transform
        self.num_classes = nclasses

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x_batch, y_batch = np.load(self.dataset[index], allow_pickle=True)
        if self.preprocessing:
            x_batch = img_processing(x_batch)
        if self.transform:
            x_batch = self.transform(x_batch)
        if self.num_classes == 2:
            y_batch = np.where(y_batch == 2, 1, y_batch)
        return x_batch, y_batch

##
# ######################
#     EVALUATION      ##
# ######################


def predict(net, data_loader, device, return_prob=True):
    predictions, labels = torch.tensor([]), torch.tensor([])
    all_prob0, all_prob1 = torch.tensor([]), torch.tensor([])
    print('\nModel prediction. Total number of steps:', len(data_loader))
    with alive_bar(len(data_loader), bar='classic', spinner='arrow') as bar:
        for _, (x, y) in enumerate(data_loader):
            x = x.to(device)
            output = net(x).detach().cpu()
            if return_prob:
                prob0 = output[:, 0]
                prob1 = output[:, 1]
                all_prob0 = torch.cat((all_prob0, prob0), dim=0)
                all_prob1 = torch.cat((all_prob1, prob1), dim=0)
            y_pred = torch.argmax(output, dim=1)
            predictions = torch.cat((predictions, y_pred), dim=0)
            labels = torch.cat((labels, y), dim=0)
            bar()
        if return_prob:
            return labels.numpy(), all_prob0.numpy(), all_prob1.numpy(), predictions.numpy()
        else:
            return labels.numpy(), predictions.numpy()


# ######################
#     PLOTTING       ##
# ######################

def plot_complete_report(data, saver_path, labels=None):
    if labels:
        data['Class'] = labels
        data = data.set_index('Class')
    plt.figure(figsize=(10,3.5))
    ax = sns.heatmap(data, annot=data, fmt='.2f', square=True, cmap=cmap)
    ax.set(ylabel='Classes')
    plt.title('Classification Report')
    plt.savefig(f'{saver_path}/Evaluation.png')
    plt.close()


def plot_auc_curve(fpr, tpr, roc_auc_1, saver_path):
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label="AUC={:.4f}".format(roc_auc_1))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('AUC curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{saver_path}/AUC.png')
    plt.close()


def plot_confusion_matrix(y_test, y_pred, saver_path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'{saver_path}/confusion_matrix.png')
    plt.close()