import numpy as np
from tqdm import tqdm
import random

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from flow_model.model import RessubFlow, save_model, load_model
from flow_model.feature_extractor import FeatureExtractor
from utils import *
from config import LOG_DIR


def set_seed(seed_value):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(c, train_loader, test_loader):
    model = RessubFlow(c)
    model.to(c.device)
    if c.checkpoint:
        model = load_model(model, c.checkpoint)

    # with open('model_structure.csv', 'w') as f:
    #     import json
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             f.write(name)
    #             f.write(';')
    #             f.write(json.dumps(param.data.shape))
    #             f.write(';')
    #             f.write(f'{param.data.view(-1).shape[0]}')
    #             f.write('\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_init, eps=c.eps, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c.t_max, eta_min=c.eta_min)

    fe = FeatureExtractor(c)
    fe.eval()
    fe.to(c.device)
    for param in fe.parameters():
        param.requires_grad = False

    det_obs = Score_Observer(c, 'DET_AUROC')
    seg_obs = Score_Observer(c, 'SEG_AUROC')

    writer = SummaryWriter(os.path.join(LOG_DIR, c.model_name))

    for epoch in range(c.epochs):
        model.train()

        train_loss = list()
        train_loss_positive = list()
        train_loss_negative = list()

        for i, (x, labels, masks, syn_x, _) in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
            optimizer.zero_grad()

            B = x.shape[0]

            x = torch.cat((x, syn_x), dim=0)
            x = x.to(c.device)

            x = fe(x)

            z, log_jac_dets = model.forward(x)
            loss, loss_positive, loss_negative = model.get_loss(B, z, log_jac_dets)

            loss.backward()
            optimizer.step()

            train_loss.append(t2np(loss))
            train_loss_positive.append(t2np(loss_positive))
            train_loss_negative.append(t2np(loss_negative))

            if epoch % 10 == 0 and i == 0:
                unorm = UnNormalize(mean=c.norm_mean, std=c.norm_std)
                syn_x_normalized = list()
                for bi in range(B):
                    syn_x_normalized.append(unorm(syn_x[bi]))
                syn_x_grid = torchvision.utils.make_grid(syn_x_normalized)

                writer.add_image('synthetic images', syn_x_grid, epoch)

        scheduler.step()

        if c.verbose:
            write_verbose(c.model_name, 'Epoch: {:d} \t train loss: {:.4f}'.format(epoch, np.mean(train_loss)))

        writer.add_scalar('train_loss_mean', np.mean(train_loss), epoch)
        writer.add_scalar('train_loss_positive_mean', np.mean(train_loss_positive), epoch)
        writer.add_scalar('train_loss_negative_mean', np.mean(train_loss_negative), epoch)
        writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)

        if epoch % 10 == 0 or epoch == c.epochs - 1:
            # evaluate
            model.eval()
            if c.verbose:
                print('\nCompute maps, loss and scores on test set:')
            test_labels = list()
            test_masks = list()
            test_flow_loss = list()

            test_det_score = list()
            test_seg_score = list()

            with torch.no_grad():
                for i, (x, labels, masks, _, _) in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                    B = x.shape[0]

                    x = x.to(c.device)
                    x = fe(x)

                    z, log_jac_dets = model.forward(x)
                    loss_flow, _, _ = model.get_loss(B, z, log_jac_dets)

                    #prob scoring
                    log_prob = model.prior.log_prob(z)

                    prob = torch.exp(log_prob)
                    prob = torch.mean(prob, dim=1, keepdim=True)
                    seg_score_prob = F.interpolate(-prob, size=c.img_size, mode='bilinear', align_corners=False)
                    test_seg_score.extend(t2np(seg_score_prob))

                    test_det_score.append(np.max(t2np(seg_score_prob), axis=(1, 2, 3)))

                    test_flow_loss.append(t2np(loss_flow))

                    test_labels.append(labels)
                    test_masks.extend(t2np(masks))

            if c.verbose:
                write_verbose(c.model_name, 'Epoch: {:d} \t test flow loss: {:.4f}'.format(epoch, np.mean(test_flow_loss)))
            writer.add_scalar('test_loss', np.mean(test_flow_loss), epoch)

            test_labels = np.concatenate(test_labels)
            image_label = np.array([0 if l == 0 else 1 for l in test_labels])

            gt_masks = np.squeeze(np.asarray(test_masks, dtype=np.bool), axis=1)

            det_score = np.concatenate(test_det_score, axis=0)
            det_score_auroc = roc_auc_score(image_label, det_score)

            seg_score = np.squeeze(np.asarray(test_seg_score, dtype=np.float32), axis=1)
            seg_score_auroc = roc_auc_score(gt_masks.flatten(), seg_score.flatten())

            if c.verbose:
                write_verbose(
                    c.model_name, 'Epoch: {:d} \t det_score: {:.4f} \t seg_score: {:.4f}'.format(
                        epoch, det_score_auroc, seg_score_auroc))

            det_obs.update(det_score_auroc, epoch, print_score=c.verbose or epoch == c.epochs - 1)
            seg_obs.update(seg_score_auroc, epoch, print_score=c.verbose or epoch == c.epochs - 1)

            writer.add_scalar('det_score_auroc', det_score_auroc, epoch)
            writer.add_scalar('seg_score_auroc', seg_score_auroc, epoch)

            if c.save_model and det_obs.max_epoch == epoch:
                save_model(model, c.model_name, 'det')
            if c.save_model and seg_obs.max_epoch == epoch:
                save_model(model, c.model_name, 'seg')

            if min(epoch - det_obs.max_epoch, epoch - seg_obs.max_epoch) > 50:
                break

    return
