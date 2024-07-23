# VoxCeleb Disentangler 

This repository contains the environment-disentangled representation learning framework for training speaker recognition models described in the paper 'Disentangled Representation Learning for Environment-agnostic Speaker Recognition'.

This repository is based on the original repository `voxceleb_trainer`([Link](https://github.com/clovaai/voxceleb_trainer)), so you can follow original process of the repository. For adeversarial-learning phase, we refer to `voxceleb_unsupervised`([Link](https://github.com/joonson/voxceleb_unsupervised)), an unsupervised learning framework based on `voxceleb_trainer`.

### Dependencies
```
pip install -r requirements.txt
```

### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training.

```
python ./dataprep.py --save_path data --download --user USERNAME --password PASSWORD 
python ./dataprep.py --save_path data --extract
python ./dataprep.py --save_path data --convert
```
In order to use data augmentation, also run:

```
python ./dataprep.py --save_path data --augment
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

### Dataset setup 

For this repository, you can follow the example setup below (Optional).
```
datasets/
└──voxceleb1/
└──voxceleb2/
└──manifests/
  -- train_list.txt
  -- veri_test2.txt
```

you can download voxceleb1-based verification set files in [here](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)

### Training examples

Our environment-disentangler is implemented based on two popular models - ResNetSE34V2 and ECAPA_TDNN.

- ResNetSE34V2 with Disentangler:
```
python ./trainSpeakerNet.py --config ./configs/ResNetSE34V2_SFAP_AE.yaml
```

- ECAPA_TDNN with Disentangler:
```
python ./trainSpeakerNet.py --config ./configs/ECAPA_SFAP_AE.yaml
```
(Our `ECAPA_TDNN` model's code is based on [here](https://github.com/clovaai/voxceleb_trainer/issues/86))

You can pass individual arguments that are defined in trainSpeakerNet.py by `--{ARG_NAME} {VALUE}`.
Note that the configuration file overrides the arguments passed via command line.

### Pretrained models

A pretrained model of `ResNetSE34V2`, can be downloaded from [here](https://github.com/kaistmm/voxceleb-disentangler/raw/main/pretrained/baseline_resnetse34V2_de.model).

The following script should return: (EER, %) `Vox1-O : 0.8457`,  `VoxSRC22 : 3.0123`, `VoxSRC23 : 5.4046`.

You will be given an option to save the scores.

```
python ./trainSpeakerNet.py --eval --config ./configs/ResNetSE34V2_SFAP_AE.yaml --initial_model pretrained/baseline_resnetse34V2_de.model --save_path exps/test --eval_frames 400
```

A pretrained model of `ECAPA_TDNN`, can be downloaded from [here](https://github.com/kaistmm/voxceleb-disentangler/raw/main/pretrained/baseline_ecapatdnn_de.model).

The following script should return: (EER, %) `Vox1-O : 0.8188`,  `VoxSRC22 : 3.0814`, `VoxSRC23 : 5.7594`.

```
python ./trainSpeakerNet.py --eval --config ./configs/ECAPA_SFAP_AE.yaml --initial_model pretrained/baseline_ecapatdnn_de.model --save_path exps/test --eval_frames 400 
```

You can access the pretrained models in `pretrained/` directory.


### Implemented loss functions
```
-- Original(`voxceleb_trainer`) --
Softmax (softmax)
AM-Softmax (amsoftmax)
AAM-Softmax (aamsoftmax)
GE2E (ge2e)
Prototypical (proto)
Triplet (triplet)
Angular Prototypical (angleproto)
-- Ours --
Mean Average Pearson's correlation (MAPC)
```

### Implemented models and encoders(pooling layers)
```
ResNetSE34L   (SAP, ASP)
ResNetSE34V2  (SAP, ASP) + Disentangler (Implemented)
VGGVox40      (SAP, TAP, MAX)
ECAPA_TDNN    (ECA) + Disentangler      (Implemented)
```

### Data augmentation

`--augment True` enables online data augmentation, described in `voxceleb_trainer`.

### Adding new models and loss functions

You can add new models and loss functions to `models` and `loss` directories respectively. See the existing definitions for examples.

### Accelerating training

- Use `--mixedprec` flag to enable mixed precision training. This is recommended for Tesla V100, GeForce RTX 20 series or later models.

- Use `--distributed` flag to enable distributed training.

  - GPU indices should be set before training using the command `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

  - If you are running more than one distributed training session, you need to change the `--port` argument.
