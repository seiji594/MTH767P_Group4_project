#
# Created by V.Sotskov on 19 March 2022
#
from collections import namedtuple

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from PIL import Image
from torch.nn.modules.activation import *
from torchvision.datasets import VisionDataset
from torch.utils.data import SubsetRandomSampler, DataLoader, BatchSampler

import matplotlib
import matplotlib.pyplot as plt

output_path = Path('./outputs').resolve()
models_path = Path('./models').resolve()

IMAGE_DIM = 48
IMAGE_CH = 1
TORCH_LAYERS = dict(conv2d=nn.Conv2d,
                    maxpool2d=nn.MaxPool2d,
                    avgpool2d=nn.AvgPool2d,
                    batchnorm2d=nn.BatchNorm2d,
                    dropout=nn.Dropout,
                    dropout2d=nn.Dropout2d,
                    linear=nn.Linear)


class EmotionsDataset(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset
        fname (string): name of the dataset file
        transform (callable, optional): A function/transform that takes in an image
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            fname: str,
            transform: Optional[Callable] = None,
            seed: int = 123456789):

        super().__init__(root, transform=transform)

        self.rng = np.random.default_rng(seed=seed)
        self.g_cpu = torch.Generator()
        self.g_cpu.manual_seed(seed)
        self.train_idxs = None
        self.data: Any = []
        self.targets = []

        path = (Path(root) / fname).resolve()
        print("Loading dataset...", end="\t")
        data = pd.read_csv(path)
        self.targets = data['emotion'].values
        x = data[' pixels'].str.split(' ', expand=True)
        x = x.astype('uint8').values
        self.data = x.reshape(-1, 1, 48, 48)  # one channel, 48x48 image
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        print("Done")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.reshape(48, 48))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def split(self, *, ratio=0.8, batch_size=1):
        """
        The function to split a dataset into train/test subsets
        Args:
            ratio (float): the ratio of train portion to test portion
            batch_size (int): size of batch to sample on each iteration
            seed (int): seed for random number generator
        """
        # define len of data
        n = len(self.data)

        # calculate the indexes
        self.train_idxs = self.rng.choice(np.arange(n), size=int(ratio * n), replace=False)
        ix_test = np.delete(np.arange(n), self.train_idxs)

        trainsmplr = SubsetRandomSampler(self.train_idxs, self.g_cpu)
        testsmplr = SubsetRandomSampler(ix_test, self.g_cpu)

        return DataLoader(self, batch_size=batch_size, sampler=trainsmplr), \
               DataLoader(self, batch_size=1, sampler=testsmplr)

    def kfold(self, K, batch_size=1):
        KfoldLoader = namedtuple('KfoldLoader', ['validation', 'holdout'])
        data_size = len(self.train_idxs)
        indexes = self.rng.permutation(data_size)
        m, r = divmod(data_size, K)
        indexes_split = [
            indexes[i * m + min(i, r):(i + 1) * m + min(i + 1, r)]
            for i in range(K)
        ]
        loaders = []
        for i in range(K):
            training_indexes = np.concatenate([indexes_split[j] for j in range(K) if (j != i)])
            valid = BatchSampler(training_indexes, batch_size=batch_size, drop_last=False)
            hold = BatchSampler(indexes_split[i], batch_size=1, drop_last=False)
            loaders.append(KfoldLoader(DataLoader(self, batch_size=batch_size, sampler=valid),
                                       DataLoader(self, batch_size=1, sampler=hold)))
        return loaders


# class SimpleNet(nn.Module):
#     def __init__(self, layers, activation, pooling=None):
#         super().__init__()
#         layer_list = []
#         for layer in layers:
#             layer_list.append(layer.pop('ltype')(**layer))
#         self.layers = nn.ModuleList(layer_list)
#         self.activation = activation
#         self.pool = pooling
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             if isinstance(self.layers[i - 1], nn.Conv2d) and isinstance(layer, nn.Linear):
#                 x = torch.flatten(x, 1)
#             x = layer(x)
#             if i != len(self.layers) - 1:
#                 x = self.activation(x)
#                 if isinstance(layer, nn.Conv2d) and self.pool is not None:
#                     x = self.pool(x)
#         return x


class ConvNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        linl_list = []
        d = IMAGE_DIM
        convl_list, inch, li = make_convpart(layers, d)
        linl_list = make_linpart(layers[li:], inch)
        self.conv = nn.Sequential(*convl_list)
        self.linear = nn.Sequential(*linl_list)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class AttentionalNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # Spatial transformer localization-network
        # self.localization = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=3),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=3),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )
        stn_layers = layers['attention']
        stn_conv, inch, li = make_convpart(stn_layers, IMAGE_DIM)
        self.localization = nn.Sequential(*stn_conv)

        # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(10 * 10 * 10, 48),
        #     nn.ReLU(True),
        #     nn.Linear(48, 3 * 2)
        # )
        stn_fc = make_linpart(stn_layers[li:], inch)
        self.fc_loc = nn.Sequential(*stn_fc)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Sequential convolution network
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 10, kernel_size=3),
        #     nn.ReLU(True),
        #     nn.Conv2d(10, 10, kernel_size=3),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(10, 10, kernel_size=3),
        #     nn.ReLU(True),
        #     nn.Conv2d(10, 10, kernel_size=3),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )
        #
        # self.conv2_drop = nn.Dropout2d()
        fex_layers = layers['features']
        fex_conv, inch, li = make_convpart(fex_layers, IMAGE_DIM)
        self.conv = nn.Sequential(*fex_conv)
        # self.fc1 = nn.Linear(810, 50)
        # self.fc2 = nn.Linear(50, 7)
        fex_fc = make_linpart(fex_layers[li:], inch)
        self.fc = nn.Sequential(*fex_fc)

    def stn(self, x):
        xs = self.localization(x)
        # xs = xs.view(-1, 10 * 10 * 10)
        xs = torch.flatten(xs, 1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        # x = self.conv1(x)
        # x = self.conv2_drop(self.conv2(x))
        x = self.conv(x)
        # x = x.view(-1, 810)
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc(x)
        return x


####
# Helper functions
###
def train(model, criterion, optimizer, scheduler, trainloader, num_epochs, verbose=True):
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Epoch {epoch + 1}")
        for i, data in tqdm(enumerate(trainloader)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999 and verbose:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        if scheduler is not None:
            if isinstance(scheduler, list):
                for s in scheduler:
                    s.step()
            else:
                scheduler.step()
    print('Finished Training')


def cross_validate(nfolds, dataset, model, criterion, optimizer, scheduler, labels_dict, batch_size):
    loaders = dataset.kfold(nfolds, batch_size=batch_size)

    validation_error = 0
    for ll in loaders:
        train(model, criterion, optimizer, scheduler, ll.validation, 1, False)
        accydf = check_accuracy(model, criterion, ll.holdout, labels_dict, False)
        accy = np.sum(np.diag(accydf.drop('avg_loss', axis=1).values)) / accydf.drop('avg_loss', axis=1).sum().sum()
        validation_error += (1 - accy)
    return validation_error / nfolds


def grid_search(objective, grid, param_name=""):
    values = np.array([])
    for point in grid:
        print(f"Testing {param_name} value {point:.6f}...")
        values = np.append(values, objective(point))
        print(f"Done, error={values[-1]:.6f}")
    return grid[np.argmin(values)]


def make_convpart(layers, dim, outch=None):
    convl_list = []
    for li, layer in enumerate(layers):
        ltype = layer['ltype'].lower()
        try:
            ltype = TORCH_LAYERS[ltype]
        except KeyError:
            e = NotImplementedError(f"Adding this type of layer ({layer['ltype']}) is not implemented")
            print(e)
            continue
        activation = layer.get('activation')
        inch = IMAGE_CH if outch is None else outch
        outch = layer.get('out_channels', outch)
        k = layer.get('kernel')
        pad = layer.get('padding', 0)
        stride = layer.get('stride', 1)
        grp = layer.get('groups', 1)
        bias = layer.get('bias', True)

        if ltype == nn.Conv2d:
            pad = 0 if pad == 'valid' else pad
            if pad != 'same':
                dim = np.floor((dim + 2 * pad - k) / stride + 1)
            convl_list.append(ltype(in_channels=inch, out_channels=outch,
                                    kernel_size=k, stride=stride, padding=pad,
                                    groups=grp, bias=bias))
        elif ltype == nn.MaxPool2d or ltype == nn.AvgPool2d:
            dim = np.floor((dim - k) / stride + 1)
            convl_list.append(ltype(kernel_size=k, stride=stride))
        elif ltype == nn.BatchNorm2d:
            mom = layer.get('momentum', 0.1)
            aff = layer.get('affine', True)
            convl_list.append(ltype(outch, momentum=mom, affine=aff))
        elif ltype == nn.Dropout or ltype == nn.Dropout2d:
            p = layer.get('p', 0.5)
            convl_list.append(ltype(p=p))
        elif ltype == nn.Linear:
            inch = int(dim * dim * outch)
            break
        else:
            continue

        if activation is not None:
            convl_list.append(activation)

    return convl_list, inch, li


def make_linpart(layers, inch):
    linl_list = []
    for layer in layers:
        # We moved to linear layers
        ltype = layer['ltype'].lower()
        try:
            ltype = TORCH_LAYERS[ltype]
        except KeyError:
            e = NotImplementedError(f"Adding this type of layer ({layer['ltype']}) is not implemented")
            print(e)
            continue
        outch = layer['out_features']
        bias = layer.get('bias', True)
        activation = layer.get('activation')
        if ltype == nn.Linear:
            linl_list.append(ltype(inch, outch, bias=bias))
        elif ltype == nn.BatchNorm2d:
            mom = layer.get('momentum', 0.1)
            aff = layer.get('affine', True)
            linl_list.append(ltype(outch, momentum=mom, affine=aff))
        elif ltype == nn.Dropout or ltype == nn.Dropout2d:
            p = layer.get('p', 0.5)
            linl_list.append(ltype(p=p))
        else:
            continue
        if activation is not None:
            linl_list.append(activation)
        inch = outch

    return linl_list


def summary(input_df):
    avg_loss = input_df['avg_loss'].iloc[0]
    df = input_df.drop('avg_loss', axis=1)
    correct = np.sum(np.diag(df.values))
    numtests = df.sum().sum()
    avg_loss = f"Average loss: {avg_loss:.4f}"
    accy = f"Accuracy: {correct}/{numtests} ({100 * correct / numtests:.0f}%)"
    cls_accy = np.diag(df.values) / df.sum(axis=1).values
    return pd.DataFrame({avg_loss: list(df.index), accy: cls_accy},
                        index=['Accuracy for class:'] * len(df.index))


def check_accuracy(model, criterion, testloader, labels_dict, verbose=True):
    labels = list(labels_dict.values())
    res = pd.DataFrame(index=labels, columns=labels).fillna(0)
    res.index.name = 'True label'
    res.columns.name = 'Predicted label'
    total = 0

    with torch.no_grad():
        numtestcases = 0
        for data in tqdm(testloader):
            numtestcases += 1
            image, label = data
            output = model(image)
            total += criterion(output, label).item()
            pred = output.max(1)[1]
            pred_lbl = labels_dict[pred.item()]
            true_lbl = labels_dict[label.item()]
            res.loc[true_lbl, pred_lbl] += 1
        total /= numtestcases

    correct = np.sum(np.diag(res.values))
    if verbose:
        print(f'Average loss: {total:.4f}\tAccuracy: {correct}/{numtestcases} ({100 * correct / numtestcases:.1f}%)')
        for lbl in labels:
            totals = res.sum(axis=1)
            disp = res / totals
            print(f'Accuracy for class {lbl:5s}: {100 * disp.loc[lbl, lbl]:.1f}%')

    return res.assign(avg_loss=total)


def parse_schedulers(sch_list):
    schedulers = sch_list if isinstance(sch_list, list) else [sch_list]
    retdict = {}
    for i, sch in enumerate(schedulers):
        scname = sch.__class__.__name__
        scparms = {k: v for k, v in sch.__dict__.items() if k[0] != "_"}
        scparms.pop('optimizer')
        scparms.pop('base_lrs')
        scparms.pop('verbose')
        retdict[f"scheduler{i + 1}"] = {'class': scname, 'params': scparms}
    return retdict


def save_model(model, modelargs, criterion, optimizer, scheduler, num_epochs, results, batch=None):
    ftypes = ['torch', 'yaml ']
    config = dict()
    mname = model.__class__.__name__
    config["model"] = {"class": mname, "params": modelargs}
    lfname = criterion.__class__.__name__
    lfparms = criterion.__dict__
    lfparms = {k: v for k, v in lfparms.items() if k != 'training' and k[0] != '_'}
    config["loss"] = {"class": lfname, "params": lfparms}
    opname = optimizer.__class__.__name__
    opparms = optimizer.__dict__['defaults']
    config["optimizer"] = {"class": opname, "params": opparms}
    config["epochs"] = num_epochs
    if scheduler is not None:
        config["lr_adjusters"] = parse_schedulers([scheduler])

    smry = summary(results)
    accy = smry.columns[1].split("(")[-1][:-2]
    fname = f"{mname.lower()}_accy{accy}_v1"
    # fname = f"{mname.lower()}_{opname.lower()}-lr{opparms['lr']}_v1"
    while (models_path / (fname + ".pth")).exists():
        v = int(fname[-1])
        fname = fname[:-1] + str(v + 1)
    torch.save(model.state_dict(), models_path / (fname + '.pth'))
    with open(models_path / (fname + '.yml'), 'w') as f:
        yaml.dump(config, f, yaml.CDumper)
    if batch is not None:
        try:
            _ = torch.onnx.export(model, batch, models_path / (fname + '.onnx'),
                                  input_names=['Image'], output_names=['Emotion label'])
            ftypes = ", ".join(ftypes) + 'and onnx'
        except RuntimeError:
            ftypes = "and ".join(ftypes)
            pass

    smry.to_latex(output_path / (fname + '.tex'), column_format='lrr', escape=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    data = results.drop('avg_loss', axis=1)
    im, bar = heatmap(100 * data / data.sum(axis=1), ax=ax, cbarlabel="accuracy, %")
    annotate_heatmap(im, valfmt="{x:.1f}%", textcolors=('w', 'b'))
    plt.savefig(output_path / (fname + '.png'), format='png')
    plt.close(fig)
    print(f"Model specs ({ftypes} files) saved into ./models folder under name {fname}.\nThe model's results (.tex and "
          f".png files) are saved into ./outputs folder under the same name")
    return fname


def load_model(fname):
    with open(models_path / (fname + ".yml"), 'r') as f:
        cfg = yaml.load(f, yaml.CLoader)
    model = cfg['model']
    invocation = "{}({})".format(model['class'], "" if model['params'] is None else model['params'])
    net = eval(invocation)
    net.load_state_dict(torch.load(models_path / (fname + ".pth")))
    return net, cfg


# def check_accuracy(model, criterion, testloader, labels_dict, save=False):
#     correct = 0
#     total = 0
#     correct_pred = {labels_dict[clss]: 0 for clss in labels_dict}
#     total_pred = {labels_dict[clss]: 0 for clss in labels_dict}
#     numtestcases = len(testloader.dataset) - len(testloader.dataset.train_idxs)
#
#     with torch.no_grad():
#         i = 0
#         for data in tqdm(testloader):
#             i += 1
#             image, label = data
#             output = model(image)
#             total += criterion(output, label).item()
#             pred = output.max(1, keepdim=True)[1]
#             iftrue = pred.eq(label.view_as(pred)).sum().item()
#             correct += iftrue
#             correct_pred[labels_dict[label.item()]] += iftrue
#             total_pred[labels_dict[label.item()]] += 1
#         total /= numtestcases
#
#     avg_loss = f"Average loss: {total:.4f}"
#     accy = f"Accuracy: {correct}/{numtestcases} ({100 * correct / numtestcases:.0f}%)"
#     cls_accy = []
#
#     for classname, correct_count in correct_pred.items():
#         accuracy = 100 * float(correct_count) / total_pred[classname]
#         cls_accy.append(f"{accuracy:.1f}%")
#
#     res = pd.DataFrame({avg_loss: list(correct_pred.keys()), accy: cls_accy},
#                        index=['Accuracy for class:'] * len(correct_pred))
#     if save:
#         fname = save if isinstance(save, str) else model.__class__.__name__
#         res.to_latex(output_path / (fname + '.tex'), column_format='lrr', escape=True)
#         print(f"Saved results table into ./outputs folder under name {fname}.tex")
#
#     return res

###
# Code from matplotlib
###
def heatmap(data, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a pandas DataFrame.

    Parameters
    ----------
    data
        A pd.DataFrame of shape (M, N).
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if cbar_kw is None:
        cbar_kw = {}

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=data.columns)
    ax.set_yticks(np.arange(data.shape[0]), labels=data.index)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.set_xlabel(data.columns.name)
    ax.set_ylabel(data.index.name)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
