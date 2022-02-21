[![arXiv](https://img.shields.io/badge/arXiv-2202.07133-b31b1b.svg)](https://arxiv.org/abs/2202.07133)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
## Sim-to-Real Lane Detection and Classification

Code for the paper [Sim-to-Real Domain Adaptation for Lane Detection and Classification in Autonomous Driving](https://arxiv.org/abs/2202.07133). 
If you use this code or our simulation [dataset](./DATASET.md), please cite our paper:
```
@misc{hu2022simtoreal,
      title={Sim-to-Real Domain Adaptation for Lane Detection and Classification in Autonomous Driving}, 
      author={Chuqing Hu and Sinclair Hudson and Martin Ethier and Mohammad Al-Sharman and Derek Rayside and William Melek},
      year={2022},
      eprint={2202.07133},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### License
Adapted work Copyright (C) 2021 Anita Hu, Sinclair Hudson, Martin Ethier.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Original work ([mingyuliutw/UNIT](https://github.com/mingyuliutw/UNIT)) Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

Original work ([cfzd/Ultra Fast Lane Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)) Copyright (c) 2020 cfzd.  All rights reserved. Licensed under the MIT license.

### Code usage

Clone this repository
```
git clone https://github.com/anita-hu/sim2real-lane-detection.git
```
Training logs are done on [Weights & Biases](https://wandb.ai/site), please install with
```
pip install wandb
```
Login to your user where the credentials will be passed into the docker container with the launch script
```
wandb login
```

#### Docker
Build docker image
```
docker build -t sim2real .
```
Launch docker container

NOTE: `/path/to/dataset_root` is a folder containing all datasets to be mounted in the docker container
```
bash start.sh /path/to/dataset_root
```

#### Datasets
Please see [DATASET.md](./DATASET.md)

#### Training
Download VGG16 weights from [Google Drive](https://drive.google.com/drive/folders/1bXOdkQjsBlMXjwDizK1TxG-GAhhJVJua?usp=sharing) and place in the following folder
```
./models/vgg16.pth
```
For training, run
```
python train.py --config configs/tusimple/ada.yaml
```
```
usage: train.py [-h] [--config CONFIG] [--output_path OUTPUT_PATH] [--resume]
                [--entity ENTITY] [--project PROJECT]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       path to the config file
  --output_path OUTPUT_PATH
                        path to create outputs folder and contains
                        models/vgg16.pth
  --resume              resume training session
  --entity ENTITY       wandb team name, set to None for default entity
                        (username)
  --project PROJECT     wandb project name [default: sim2real-lane-detection]
```

#### Two-stage training

For reproducing the "Two Stage" training results, first, a translator must be trained, with a special config that doesn't use lane losses:
```
python train.py --config configs/tusimple/unit_s2r.yaml
```

Next,`translate_dataset.py` copies a whole dataset,
translates every image using a trained translator,
and saves it to be used in future runs:

```
mkdir outputs/ds
chmod 777 outputs/ds
python translate_dataset.py  --config configs/tusimple/unit_s2r.yaml  \
--checkpoint_dir outputs/{run_id}/checkpoints/ \
--new_data_folder outputs/ds/
```

Finally, once the dataset has been translated, the location can be referenced in
a normal training configuration file, and training occurs as normal:

```
python train.py --config configs/tusimple/s2r_baseline.yaml 
```

where `s2r_baseline.yaml` has:

```yaml
...
dataset: TuSimple                  
dataA_root: outputs/ds/WATO_TuSimple
```

#### Evaluating
Within the docker container, build the evaluation tool
```
cd evaluation/culane
make
```
For evaluation, run
```
python test.py --config configs/tusimple/unit.yaml \
               --checkpoint outputs/{run_id}/checkpoints/gen.pt \
               --output_folder results
```

#### Demo

You can visualize results with openCV with the `demo.py` script:

```
python demo.py --config configs/tusimple/unit.yaml \
               --checkpoint /path/to/pretrained/gen.pt
```
Find the outputs in the `outputs/` folder.

### Reference

This code is built upon [UNIT](https://github.com/mingyuliutw/UNIT) and 
[Ultra Fast Lane Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) github repositories. 

Please cite their papers:

[Unsupervised Image-to-Image Translation Networks (NeurIPS 2017)](https://proceedings.neurips.cc/paper/2017/hash/dc6a6489640ca02b0d42dabeb8e46bb7-Abstract.html)

[Ming-Yu Liu](http://mingyuliu.net/), [Thomas Breuel](http://www.tmbdev.net/), and [Jan Kautz](http://jankautz.com/)
```
@inproceedings{NIPS2017_dc6a6489,
 author = {Liu, Ming-Yu and Breuel, Thomas and Kautz, Jan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Unsupervised Image-to-Image Translation Networks},
 url = {https://proceedings.neurips.cc/paper/2017/file/dc6a6489640ca02b0d42dabeb8e46bb7-Paper.pdf},
 volume = {30},
 year = {2017}
}
```
[Multimodal Unsupervised Image-to-Image Translation (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.html)

[Xun Huang](http://www.cs.cornell.edu/~xhuang/), [Ming-Yu Liu](http://mingyuliu.net/), 
[Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/), and [Jan Kautz](http://jankautz.com/), 
```
@InProceedings{Huang_2018_ECCV,
author = {Huang, Xun and Liu, Ming-Yu and Belongie, Serge and Kautz, Jan},
title = {Multimodal Unsupervised Image-to-image Translation},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```
[Ultra Fast Structure-aware Deep Lane Detection (ECCV 2020)](https://link.springer.com/chapter/10.1007%2F978-3-030-58586-0_17)

[Zequn Qin](https://scholar.google.com/citations?user=D-Wyao4AAAAJ), [Huanyu Wang](https://scholar.google.com/citations?user=711Ww7gAAAAJ), and [Xi Li](https://scholar.google.ca/citations?user=TYNPJQMAAAAJ)
```
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}
```
