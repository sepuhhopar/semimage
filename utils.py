from copy import deepcopy

import cv2
import numpy as np
import torch
import os.path as path

from sklearn.metrics import f1_score
from torch.nn.functional import binary_cross_entropy, conv2d


def create_edges(image, threshold, as_binary):
    if not as_binary:
        image[image == 0] = 0.0
        image[np.logical_and(image > 0, image < threshold)] = 2.0
        image[image >= threshold] = 1.0
    else:
        image = np.where(image >= threshold, 1.0, 0.0)

    return image


class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root='data/HED-BSDS',
                 split='test',
                 threshold=0.3,
                 normalize=False,
                 resize_function=None,
                 binary=False,
                 transform=None,
                 reduce_labels=False):

        super(EdgeDataset, self).__init__()

        self.binary = binary
        self.root = root
        self.split = split
        self.transform = transform
        self.reduce_labels = reduce_labels
        self.threshold = threshold * 256
        self.resize_f = resize_function
        self.normalize = normalize

        if self.split == 'train':
            self.file_list = path.join(self.root, 'train_rgb.lst')
        elif self.split == 'test':
            self.file_list = path.join(self.root, 'test_rgb.lst')
        else:
            raise ValueError('Invalid split type!')

        with open(self.file_list, 'r') as f:
            self.file_list = f.readlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        img_file, label_file = self.file_list[index].split()

        label = cv2.imread(path.join(self.root,
                                     'imgs',
                                     self.split,
                                     'label',
                                     label_file))

        img = cv2.imread(path.join(self.root,
                                   'imgs',
                                   self.split,
                                   'image',
                                   img_file))

        if label.ndim == 3:
            label = np.squeeze(label[:, :, 0])

        # if not self.binary:
        #     label[label == 0] = 0.0
        #     label[np.logical_and(label > 0, label < self.threshold)] = 2.0
        #     label[label >= self.threshold] = 1.0
        # else:
        #     # mask = np.logical_and(label > 0, label < self.threshold)
        #     label = np.where(label >= self.threshold, 1.0, 0.0)

        label = create_edges(label, self.threshold, self.binary)
        if self.normalize:
            img = img / 255.0

        img = np.float32(img)
        label = np.float32(label)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label)

        if self.resize_f is not None:
            img = self.resize_f(img)
            label = self.resize_f(label[None])[0]

        return img, label


# def ois(prediction, edges, fixed_threshold=None):
#     assert len(prediction) == len(edges)
#     assert prediction.ndim == edges.ndim
#
#     if len(prediction.shape) == 2:
#         prediction = prediction[None]
#         edges = edges[None]
#
#     if fixed_threshold is not None:
#         thresholds = [fixed_threshold]
#     else:
#         thresholds = np.linspace(1e-3, 1.0, 100, endpoint=False)
#
#     all_scores = []
#     all_thresholds = []
#     best_predictions = []
#
#     for p, e in zip(prediction, edges):
#         best_threshold = None
#         best_score = None
#
#         for th in thresholds:
#             pred = deepcopy(p)
#
#             pred[pred == 0] = 0
#             pred[np.logical_and(pred > 0, pred < th)] = 2
#             pred[pred >= th] = 1
#
#             # pred[pred >= th] = 1
#             # pred[pred < th] = 0
#
#             # score = (pred == e).mean()
#             if pred.sum() == 0:
#                 score = 0
#             else:
#                 score = f1_score(e.reshape(-1), pred.reshape(-1),
#                                  average='micro')
#
#             score *= 100
#
#             if best_threshold is None or score > best_score:
#                 best_score = score
#                 best_threshold = th
#                 best_predictions.append(pred)
#
#         all_scores.append(best_score)
#         all_thresholds.append(best_threshold)
#
#     return all_scores, all_thresholds, best_predictions

@torch.no_grad()
def ois(prediction, edges, as_binary, fixed_threshold=None):
    assert len(prediction) == len(edges)
    assert prediction.ndim == edges.ndim

    if len(prediction.shape) == 2:
        prediction = prediction[None]
        edges = edges[None]

    if fixed_threshold is not None:
        thresholds = [fixed_threshold]
    else:
        thresholds = np.linspace(1e-3, 1.0, 100, endpoint=False)

    all_scores = []
    all_thresholds = []
    best_predictions = []

    edges = np.asarray(edges, dtype=int)

    for p, e in zip(prediction, edges):
        best_threshold = None
        best_score = None

        for th in thresholds:
            pred = deepcopy(p)

            # pred[pred == 0] = 0
            # pred[np.logical_and(pred > 0, pred < th)] = 2
            # pred[pred >= th] = 1
            pred = create_edges(pred, th, as_binary)
            pred = np.asarray(pred, dtype=int)

            if pred.sum() == 0:
                score = 0
            else:
                score = f1_score(e.reshape(-1), pred.reshape(-1),
                                 average='micro')

            score *= 100

            if best_threshold is None or score > best_score:
                best_score = score
                best_threshold = th
                best_predictions.append(pred)

        all_scores.append(best_score)
        all_thresholds.append(best_threshold)

    return all_scores, all_thresholds, best_predictions


@torch.no_grad()
def ods_score(model, dataloader, as_binary):
    device = next(model.parameters()).device

    thresholds = np.linspace(1e-3, 1.0, 100, endpoint=False)
    best_threshold = None
    best_score = None

    for th in thresholds:
        all_scores = []
        for x, y in dataloader:
            x = x.to(device).float()
            prediction = model(x)

            prediction = prediction.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            scores, thresholds, _ = ois(prediction, y, fixed_threshold=th,
                                        as_binary=as_binary)
            all_scores.extend(scores)

        score = np.mean(all_scores)
        if best_threshold is None or score > best_score:
            best_score = score
            best_threshold = th

    return best_score, best_threshold


def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = binary_cross_entropy(
        prediction, labelf, weight=mask, reduction='sum')

    return cost


def bdrloss(prediction, label, radius):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''
    device = prediction.device

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

    texture_mask = conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return torch.sum(cost.float().mean((1, 2, 3)))


def textureloss(prediction, label, mask_radius):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    device = prediction.device
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss.float().mean((1, 2, 3)))


def cats_loss(prediction, label, l_weight=[0.,0.]):
    # tracingLoss

    device = prediction.device
    tex_factor,bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    prediction = torch.sigmoid(prediction)

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduction='none')
    cost = torch.sum(cost.float().mean((1, 2, 3)))  # by me
    label_w = (label != 0).float()
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4)

    return cost + bdr_factor * bdrcost + tex_factor * textcost


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss modified in DexiNed

    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost