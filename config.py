import os
import datetime
import argparse

VIZ_DIR = './viz'
LOG_DIR = './logdir'
RESULT_DIR = './results'
WEIGHT_DIR = './weights'
VERBOSE_DIR = './verbose'
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(VERBOSE_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MVTec', type=str, help='MVTec, BTAD, DAGM')
    parser.add_argument('--class-name', default='bottle', type=str)

    parser.add_argument('--checkpoint', default='', type=str)

    # data setting
    parser.add_argument('--few-shot', default=0, type=int, help='0 = all dataset')
    parser.add_argument('--repeat', default=1, type=int, help='repeat dataset')

    # training setting
    parser.add_argument('--feature-extractor', default='cait', type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=500, type=int)

    parser.add_argument('--coupling', default=5, type=int, help='n coupling layers')
    parser.add_argument('--hidden-channels', default=128, type=int)
    parser.add_argument('--clamp', default=2.0, type=float)
    parser.add_argument('--drop-prob', default=0.2, type=float)
    parser.add_argument('--num-heads', default=4, type=int)

    parser.add_argument('--negative-val', default=1e5, type=float)
    parser.add_argument('--neg-loss-weight', default=0.5, type=float, help='(1-weight)*positive + weight*negative')

    # optimizer parameters
    parser.add_argument('--lr-init', default=2e-4, type=float)
    parser.add_argument('--eta-min', default=1e-6, type=float)
    parser.add_argument('--t-max', default=100, type=int)
    parser.add_argument('--eps', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-5, type=float)
    parser.add_argument('--seed', default=923874273, type=int)

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--show-tqdm-bar', action='store_true')

    parser.add_argument('--murmur', default='', type=str)

    args = parser.parse_args()

    return args


def get_init_parameters(c):
    # data settings
    if c.dataset == 'MVTec':
        c.dataset_path = '/data2/liling/dataset/MVTec'  # parent directory of datasets
        c.normal_type = 'good'
    elif c.dataset == 'BTAD':
        c.dataset_path = '/data2/liling/dataset/BTAD/BTech_Dataset_transformed'  # parent directory of datasets
        c.normal_type = 'ok'
    elif c.dataset == 'DAGM':
        c.dataset_path = '/data2/liling/dataset/DAGM2007/DAGM_Transformed'
        c.normal_type = 'good'
    else:
        raise NotImplementedError('dataset must be MVTec, BTAD.')

    run_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    c.model_name = F'{c.dataset}_few{c.few_shot}_{c.feature_extractor}_{c.class_name}_{run_date}'

    # transformation settings
    c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    c.feature_extractor = 'cait_m48_448'
    c.img_size = (448, 448)
    c.img_dims = [3] + list(c.img_size)
    c.channels = 768
    c.scales = 16

    # output settings
    c.verbose = True
    c.save_model = True

    c.hide_tqdm_bar = not c.show_tqdm_bar
