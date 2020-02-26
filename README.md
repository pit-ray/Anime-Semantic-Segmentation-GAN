# Anime-Semantic-Segmentation-GAN
This architecture is based on arXiv:1802.07934, 2018. It was implemented to perform semantic segmentation for pixiv anime illust.

The Details of this architecture exist https://www.pit-ray.com/entry/semi-seg.

# Environment
|OS|Windows10 Home|
|CPU|AMD Ryzen 2600|
|GPU|MSI GTX 960 4GB|
|language|Python 3.7.1|
|framework|Chainer 7.0.0, cupy-cuda91 5.3.0|

# Result
<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/p/pit-ray/20200124/20200124213414.jpg"></img>
This result is obtained by training by Pretrained-ResNet101-DeepLab-v3.

Additionaly, parameters of the upper rusult is almost same as default value of options.py.

# How to train
You create 'dataset' directory and prepair dataset. Next, you set dataset path to option of command.
Example)
`Python3 train.py --dataset_dir dataset/example --unlabel_dataset_dir dataset/unlabel_example`
