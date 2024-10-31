import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ldc_model import LDC
from utils import EdgeDataset, ois, ods_score, cross_entropy_loss_RCF, \
    cats_loss, bdcn_loss2

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', type=str,
                    default='SEM-20230425T105517Z-001/SEM',
                    help='root of the SEM dataset')

parser.add_argument('--pretrained_path', type=str, default=None,
                    help='path to the pretrained model')

parser.add_argument('--plot', action='store_true',
                    help='if to plot test images')
parser.add_argument('--plot_dir', type=str, default=None,
                    help='plot directory')
parser.add_argument('--fine_tune',  action='store_true',
                    help='if the model must be fine tuned')

parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='the path where to save the trained model '
                         '(if empty the model won t be saved)')
parser.add_argument('--checkpoint_name',  type=str, default=None,
                    help='the save name pf the trained model')

parser.add_argument('--device', type=str, default='cpu',
                    help='gpus available')

parser.add_argument('--epochs', type=int, default=20,
                    help='number of total epochs to run')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate for all weights')
parser.add_argument('--threshold', type=float, default=0.3,
                    help='threshold to determine the ground truth (the eta parameter in the paper)')

args = parser.parse_args()

device = args.device

if torch.cuda.is_available() and device != 'cpu':
    device = 'cuda:{}'.format(device)
    torch.cuda.set_device(device)
else:
    device = 'cpu'

criterion1 = cats_loss #bdcn_loss2
criterion2 = bdcn_loss2#cats_loss#f1_accuracy2
criterion = [criterion1, criterion2]

l_weight = [[0.05, 2.], [0.05, 2.], [0.05, 2.],
            [0.1, 1.], [0.1, 1.], [0.1, 1.],
            [0.01, 4.]]  # for cats loss

tt = transforms.Compose([
    transforms.ToTensor(),
])

as_binary = not args.negative_labeling
if not as_binary:
    raise 'Now only binary edges are supported'

train_dataset = EdgeDataset(root=args.dataset_root, normalize=False,
                            split='train', transform=tt, binary=as_binary,
                            resize_function=transforms.Resize((512, 512)),
                            threshold=args.threshold)

test_dataset = EdgeDataset(root=args.dataset_root, normalize=False,
                           split='test', transform=tt, binary=as_binary,
                           resize_function=transforms.Resize((512, 512)),
                           threshold=args.threshold)

model = LDC().to(device)

if args.pretrained_path is not None:
    model.load_state_dict(torch.load(args.pretrained_path,
                                     map_location=device))

optimizer = Adam(model.parameters(), lr=args.lr)

if args.fine_tune:
    for epoch in range(10):
        losses = []
        model.train()
        for x, y in DataLoader(train_dataset, batch_size=12):
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds_list = model(x)

            loss = sum([criterion1(preds, y, l_w) for preds, l_w in
                        zip(preds_list, l_weight)])  # cats_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f'Average loss on epoch {epoch}: {np.mean(losses)}')

        with torch.no_grad():

            all_preds = []
            all_true = []

            for x, y in DataLoader(test_dataset, batch_size=5):
                x = x.to(device)
                y = y.to(device).long()

                prediction = model(x)

                prediction = torch.stack(prediction, 0).mean(0)
                prediction = torch.sigmoid(prediction).view(-1).cpu().numpy()

                prediction = np.where(prediction > args.threshold, 1, 0)

                all_true.extend(y.cpu().view(-1).tolist())
                all_preds.extend(list(prediction))

            f1 = f1_score(all_true, all_preds, average='binary')
            print(f'F1 score on test set after epoch {epoch} using the dataset threshold: '
                  f'{(f1 * 100):.2f}')

    if args.checkpoint_path is not None:
        output_dir_a = os.path.join(args.checkpoint_path)
        os.makedirs(args.checkpoint_path, exist_ok=True)

        model_path = os.path.join(args.checkpoint_path, args.checkpoint_name)
        torch.save(model.state_dict(), model_path)

with torch.no_grad():
    model.eval()

    if args.plot:
        output_dir_a = os.path.join(args.plot_dir, 'average')
        os.makedirs(output_dir_a, exist_ok=True)
        offset = 0

        for x, y in DataLoader(test_dataset, batch_size=5):
            x = x.to(device)
            y = y.to(device)

            image_shape = [_x.shape for _x in x]
            file_names = [f'img_{i + offset}.png' for i in range(len(x))]
            offset += len(file_names)

            prediction = model(x)

            prediction = torch.stack(prediction, 0).mean(0)
            prediction = torch.sigmoid(prediction).squeeze(1)
            prediction = prediction.cpu().detach().numpy()
            prediction = prediction

            images = x.cpu().detach().numpy().transpose(0, 2, 3, 1) / 255
            y_np = y.cpu().detach().numpy()

            for _p, _im, _y, fn in zip(prediction, images, y_np, file_names):
                output_file_name_a = os.path.join(output_dir_a, fn)

                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
                ax1.imshow(_im)
                ax2.imshow(_p)
                ax3.imshow(_y)

                fig.savefig(output_file_name_a)
                plt.close(fig)
                plt.show()

    all_preds = []
    all_true = []

    for x, y in DataLoader(test_dataset, batch_size=5):
        x = x.to(device)
        y = y.to(device).long()

        prediction = model(x)

        prediction = torch.stack(prediction, 0).mean(0)
        prediction = torch.sigmoid(prediction).view(-1).cpu().numpy()

        prediction = np.where(prediction > args.threshold, 1, 0)

        all_true.extend(y.cpu().view(-1).tolist())
        all_preds.extend(list(prediction))

    f1 = f1_score(all_true, all_preds, average='binary')
    print(f'Final F1 score on test set using the dataset threshold: '
          f'{(f1 * 100):.2f}')

# with torch.no_grad():
#     all_scores = []
#
#     for x, y in DataLoader(test_dataset, batch_size=5):
#         x = x.to(device).float()
#         prediction = model(x)
#
#         prediction = prediction.squeeze().cpu().numpy()
#         y = y.squeeze().cpu().numpy()
#         scores, thresholds, _ = ois(prediction, y, as_binary=as_binary)
#         all_scores.extend(scores)
#
#     print(f'OIS F1 test score: '
#           f'{np.mean(all_scores):.2f}%')
#
#     score, th = ods_score(model,
#                           DataLoader(test_dataset,
#                                      batch_size=12),
#                           as_binary=as_binary)
#
#     print(f'ODS F1 test score: '
#           f'{score:.2f}%')
