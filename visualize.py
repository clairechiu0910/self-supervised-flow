import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from config import VIZ_DIR


def viz_det_roc(c, anomaly_score, test_labels, label_types):
    img_dir = os.path.join(VIZ_DIR, c.model_name, 'viz_roc')
    os.makedirs(img_dir, exist_ok=True)

    def export_roc(c, anomaly_score, test_labels, export_name='all'):
        # Compute ROC curve and ROC area for each class
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])
        fpr, tpr, _ = roc_curve(is_anomaly, anomaly_score)
        roc_auc = auc(fpr, tpr)

        plt.clf()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class ' + c.class_name)
        plt.legend(loc="lower right")
        plt.axis('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.savefig(os.path.join(img_dir, export_name + '.png'))

    export_roc(c, anomaly_score, test_labels)
    for idx in range(1, len(label_types)):
        # combine good & one anomaly type
        filtered_indices = np.concatenate([np.where(test_labels == 0)[0], np.where(test_labels == idx)[0]])
        anomaly_score_filtered = anomaly_score[filtered_indices]
        test_labels_filtered = test_labels[filtered_indices]
        export_roc(c, anomaly_score_filtered, test_labels_filtered, export_name=label_types[idx])


def viz_det_histogram(c, anomaly_score, test_labels, label_types, thresh=2, n_bins=64):
    img_dir = os.path.join(VIZ_DIR, c.model_name, 'viz_histogram')
    os.makedirs(img_dir, exist_ok=True)

    anomaly_score[anomaly_score > thresh] = thresh

    def export_histogram(c, anomaly_score, test_labels, label_types, n_bins=64, export_name='all'):
        x_max = anomaly_score.max()
        bins = np.linspace(np.min(anomaly_score), np.max(anomaly_score), n_bins)

        plt.clf()

        for idx in range(len(label_types)):
            scores_plot = anomaly_score[test_labels == idx]
            plt.hist(scores_plot, bins, alpha=0.5, density=False, label=label_types[idx], edgecolor="black")

        ticks = np.linspace(0.5, x_max, 5)
        labels = [str(i) for i in ticks[:-1]] + ['>' + str(x_max)]
        plt.xticks(ticks, labels=labels)
        plt.xlabel(f'Detection Anomaly Score ({c.class_name})')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y')
        plt.savefig(os.path.join(img_dir, export_name + '.png'), bbox_inches='tight', pad_inches=0)

    export_histogram(c, anomaly_score, test_labels, label_types)
    for idx in range(1, len(label_types)):
        # combine good & one anomaly type
        filtered_indices = np.concatenate([np.where(test_labels == 0)[0], np.where(test_labels == idx)[0]])
        anomaly_score_filtered = anomaly_score[filtered_indices]
        test_labels_filtered = test_labels[filtered_indices]
        export_histogram(c, anomaly_score_filtered, test_labels_filtered, label_types, export_name=label_types[idx])


def viz_testing_result(c, test_images, test_labels, test_masks, det_scores, seg_scores, label_types, threshold, tag):
    print(f'Exporting testing result ({tag})...')
    image_dirs = os.path.join(VIZ_DIR, c.model_name, f'testing_result_{tag}')
    os.makedirs(image_dirs, exist_ok=True)
    score_min = np.min(seg_scores)
    score_max = np.max(seg_scores)

    for i in tqdm(range(len(test_images)), disable=c.hide_tqdm_bar):
        test_image = denormalization(test_images[i], c.norm_mean, c.norm_std)
        test_label = test_labels[i]
        test_mask = test_masks[i].transpose(1, 2, 0)
        det_score = det_scores[i]
        seg_score = seg_scores[i]
        if tag != 'ori':
            seg_score[seg_score < threshold] = 0

        fig_img, ax_img = plt.subplots(1, 3)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['bottom'].set_visible(False)
            ax_i.spines['left'].set_visible(False)
        #
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        ax_img[0].imshow(test_image)
        ax_img[1].imshow(test_mask, vmin=0, vmax=1, cmap='gray')
        ax_img[2].imshow(seg_score, vmin=score_min, vmax=score_max, cmap='Reds')
        image_file = os.path.join(image_dirs, '{:010.6f}_{:03d}_{}'.format(det_score, i, label_types[test_label]) + '.png')
        fig_img.savefig(image_file, format='png', bbox_inches='tight', pad_inches=0.0)
        plt.close()


def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x
