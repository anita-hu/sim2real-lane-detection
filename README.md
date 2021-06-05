[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
## Sim2real Lane Detection and Classification

### TODO: License

License from UNIT:

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Code usage

Model weights are stored using Git LFS. Install [Git LFS](https://git-lfs.github.com/) and clone this repository
```
git clone https://github.com/anita-hu/sim2real-lane-detection.git
```


#### Docker
Build docker image
```
docker build -t sim2real .
```
Launch docker image
```
bash start.sh
```

#### Training
```
python train.py --trainer UNIT --config configs/unit_sample_config.yaml
```

### Reference

This code is built upon [UNIT](https://github.com/mingyuliutw/UNIT), [DRIT](https://github.com/HsinYingLee/DRIT), and 
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

[DRIT++: Diverse Image-to-Image Translation via Disentangled Representations (IJCV 2020)](https://link.springer.com/article/10.1007/s11263-019-01284-z)

[Hsin-Ying Lee](http://vllab.ucmerced.edu/hylee/), [Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/), 
[Qi Mao](https://sites.google.com/view/qi-mao/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), 
[Yu-Ding Lu](https://jonlu0602.github.io/), [Maneesh Kumar Singh](https://scholar.google.com/citations?user=hdQhiFgAAAAJ), 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)
```
@article{DRIT_plus,
  author = {Lee, Hsin-Ying and Tseng, Hung-Yu and Mao, Qi and Huang, Jia-Bin and Lu, Yu-Ding and Singh, Maneesh Kumar and Yang, Ming-Hsuan},
  title = {DRIT++: Diverse Image-to-Image Translation viaDisentangled Representations},
  journal={International Journal of Computer Vision},
  pages={1--16},
  year={2020}
}
```
[Diverse Image-to-Image Translation via Disentangled Representations (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Hsin-Ying_Lee_Diverse_Image-to-Image_Translation_ECCV_2018_paper.html)

[Hsin-Ying Lee](http://vllab.ucmerced.edu/hylee/), [Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/), 
[Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), [Maneesh Kumar Singh](https://scholar.google.com/citations?user=hdQhiFgAAAAJ), 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)
```
@inproceedings{DRIT,
  author = {Lee, Hsin-Ying and Tseng, Hung-Yu and Huang, Jia-Bin and Singh, Maneesh Kumar and Yang, Ming-Hsuan},
  booktitle = {European Conference on Computer Vision},
  title = {Diverse Image-to-Image Translation via Disentangled Representations},
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