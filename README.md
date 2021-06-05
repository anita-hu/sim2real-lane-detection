[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
## Sim2real Lane Detection and Classification

### TODO: License

License from UNIT:

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Code usage

#### Docker
Build docker image
```
docker build -t sim2real .
```
Launch docker image
```
bash start.sh
```
Training
```
python train.py --trainer UNIT --config configs/unit_sample_config.yaml
```

### Reference

This code is built upon [UNIT](https://github.com/mingyuliutw/UNIT), [DRIT](https://github.com/HsinYingLee/DRIT), and 
[Ultra Fast Lane Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) github repositories. 

To cite their papers:

[UNIT](https://proceedings.neurips.cc/paper/2017/hash/dc6a6489640ca02b0d42dabeb8e46bb7-Abstract.html)
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
[MUNIT](https://openaccess.thecvf.com/content_ECCV_2018/html/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.html)
```
@InProceedings{Huang_2018_ECCV,
author = {Huang, Xun and Liu, Ming-Yu and Belongie, Serge and Kautz, Jan},
title = {Multimodal Unsupervised Image-to-image Translation},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```
