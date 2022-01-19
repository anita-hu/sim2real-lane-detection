[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
## Sim2real Lane Detection and Classification

### TODO: License

License from UNIT:

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

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
python train.py --config configs/tusimple/unit.yaml
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
               --checkpoint outputs/unit/checkpoints/gen.pt \
               --output_folder results
```

#### Demo

You can visualize results with openCV with the `demo.py` script:

```
python demo.py --config configs/tusimple/unit.yaml --checkpoint /path/to/pretrained/model.pt
```
Find the outputs in the `outputs` folder.

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
