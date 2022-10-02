import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from config import WEIGHT_DIR, get_args, get_init_parameters
from flow_model.model import RessubFlow, load_model
from flow_model.feature_extractor import FeatureExtractor
from dataset.dataset import prepare_dataloaders
from visualize import viz_det_histogram, viz_det_roc, viz_testing_result
from utils import *

MVTEC_CLASS_NAMES = [
    'carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw',
    'toothbrush', 'transistor', 'zipper'
]


def evaluate(c, model, test_loader):
    fe = FeatureExtractor(c)
    fe.eval()
    fe.to(c.device)

    sigmoid = torch.nn.Sigmoid()

    model.eval()
    if c.verbose:
        print('\nCompute maps, loss and scores on test set: {}\n'.format(c.class_name))
    test_images = list()
    test_labels = list()
    test_masks = list()
    test_flow_loss = list()
    test_det_score = list()
    test_seg_score = list()

    with torch.no_grad():
        for i, (x, labels, masks, _, _) in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            test_images.extend(t2np(x))
            test_masks.extend(t2np(masks))
            test_labels.append(labels)

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
            seg_score = sigmoid(seg_score_prob)
            test_seg_score.extend(t2np(seg_score))

            det_score = np.max(t2np(seg_score), axis=(1, 2, 3))
            test_det_score.append(det_score)

            test_flow_loss.append(t2np(loss_flow))

        test_labels = np.concatenate(test_labels)
        image_label = np.array([0 if l == 0 else 1 for l in test_labels])

        det_score = np.concatenate(test_det_score, axis=0)
        det_auroc = roc_auc_score(image_label, det_score)

        seg_score = np.squeeze(np.asarray(test_seg_score, dtype=np.float32), axis=1)
        gt_masks = np.squeeze(np.asarray(test_masks, dtype=np.bool), axis=1)
        seg_auroc = roc_auc_score(gt_masks.flatten(), seg_score.flatten())

        print('DET_AUROC: {:.2f}\t SEG_AUROC: {:.2f}\t for {}\n'.format(det_auroc * 100, seg_auroc * 100, c.class_name))

        label_types = test_loader.dataset.label_types
        viz_det_histogram(c, det_score, test_labels, label_types)
        viz_det_roc(c, det_score, test_labels, label_types)
        viz_testing_result(c, test_images, test_labels, test_masks, det_score, seg_score, label_types, 0, 'ori')


for class_name in MVTEC_CLASS_NAMES:
    c = get_args()
    c.dataset = 'MVTec'
    c.class_name = class_name
    c.checkpoint = os.path.join(WEIGHT_DIR, f'{c.dataset}_few0_{c.class_name}')

    get_init_parameters(c)

    print("evalute on checkpoint: {}\n".format(c.checkpoint))

    train_loader, test_loader = prepare_dataloaders(c)

    model = RessubFlow(c)
    model.to(c.device)
    model = load_model(model, c.checkpoint)

    evaluate(c, model, test_loader)
