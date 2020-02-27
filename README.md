# Anime-Semantic-Segmentation-GAN
This architecture is based on arXiv:1802.07934, 2018.<br>
It was implemented to perform semantic segmentation for pixiv anime illust.<br>

The Details of this architecture exist <a href="https://www.pit-ray.com/entry/semi-seg" target="_blank">my blog</a> in Japanese.

# Environment
||details|
|---|---|
|OS|Windows10 Home|
|CPU|AMD Ryzen 2600|
|GPU|MSI GTX 960 4GB|
|language|Python 3.7.1|
|framework|Chainer 7.0.0, cupy-cuda91 5.3.0|

# Result
<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/p/pit-ray/20200124/20200124213414.jpg"></img><br>
This result is obtained by training by Pretrained-ResNet101-DeepLab-v3 and it is output of unannotated anime illust.

Additionaly, parameters of the upper rusult is almost same as default value of options.py.

# How to train
You create 'dataset' directory and prepare dataset. Next, you set dataset path to option of command.<br>
Example) <br>
`Python3 train.py --dataset_dir dataset/example --unlabel_dataset_dir dataset/unlabel_example`<br>
