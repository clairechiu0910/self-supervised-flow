## Self-Supervised Normalizing Flows for Image Anomaly Detection and Localization

### Abstract 
Image anomaly detection aims to detect out-of-distribution instances. Most existing methods treat anomaly detection as an unsupervised task because anomalous training data and labels are usually scarce or unavailable. Recently, image synthesis has been used to generate anomalous samples which deviate from normal sample distribution for model training. By using the synthesized anomalous training samples, we present a novel self-supervised normalizing flow-based density estimation model, which is trained by maximizing the likelihood of normal images and minimizing the likelihood of synthetic anomalous images. By adding constraints to abnormal samples in our loss function, our model training is focused on normal samples rather than synthetic samples. Moreover, we improve the transformation subnet of the affine coupling layers in our flow-based model by dynamic stacking convolution and self-attention blocks. We evaluate our method on MVTec-AD, BTAD and DAGM datasets and achieve state-of-art performance on both the anomaly detection and localization tasks.

### Get Started
#### Environment
Python: 3.6
Install all packages with:
```
$ pip install -r requirements.txt
```

#### Datasets
We support [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [BTAD](https://www.kaggle.com/thtuan/btad-beantech-anomaly-detection), and [DAGM](https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection) for image anomaly detection and localization.
For `BTAD` and `DAGM` datasets, please reformat the dataset into the data structure as follow:
```
c.dataset_path:
    c.class_name:
        ground_truth:
            defect_type_1
            defect_type_2
            ...
        test:
            c.normal_type
            defect_type_1
            defect_type_2
            ...
        train:
            c.normal_type
```
For every dataset, `c.dataset_path`, `c.class_name`, and `c.normal_type` can be adjusted in `config.py`.

### Training
Please set the `c.dataset_path` in `config.py` before training models.  
Use `main.py` to train models for different datasets and classes.
```
python main.py --dataset=DATASET --class-name=CLASS_NAME 
```

### Testing Pretrained Models
The pretrained weights can be downloaed [here](https://drive.google.com/drive/folders/13is_aUdZBi7iZl8IgLVrA6Zs_ia1GoeX?usp=sharing).
Run `evaluate.py` for classes in MVTec-AD.
```
python evaluate.py
```

### Credits
We used `self_supervised_tasks` proposed in [NSA](https://github.com/hmsch/natural-synthetic-anomalies) to synthesize our synthetic anomaly images for our self-supervised model.