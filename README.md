# Anime-Semantic-Segmentation-GAN

## Method
This architecture is based on <a href="https://arxiv.org/abs/1802.07934">arXiv:1802.07934, 2018.</a><br>
It was implemented to perform semantic segmentation for pixiv anime illust.<br>

This GAN architectures are placed in generator.py and discriminator.py, training architecture are been updater.py or loss.py, and hyper-parameter is been options.py.

The details of this architecture exist <a href="https://www.pit-ray.com/entry/semi-seg" target="_blank">my blog</a> in Japanese.


## Result
<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/p/pit-ray/20200124/20200124213414.jpg"></img><br>
This result is obtained by training by Pretrained-ResNet101-DeepLab-v3 and it is output of unannotated anime illust.

Additionaly, parameters of the upper rusult is almost same as default value of options.py.

## How to train
Please create 'dataset' directory and prepare dataset. Next, you can set dataset path to option of command.<br>
Example) <br>
`Python3 train.py --dataset_dir dataset/example --unlabel_dataset_dir dataset/unlabel_example`<br>

## Environment
||details|
|---|---|
|OS|Windows10 Home|
|CPU|AMD Ryzen 2600|
|GPU|MSI GTX 960 4GB|
|language|Python 3.7.1|
|framework|Chainer 7.0.0, cupy-cuda91 5.3.0|

## References  
[1] Huikai Wu, Junge Zhang, Kaiqi Huang, Kongming Liang, Yizhou Yu. FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation. <i>arXiv preprint <a href="https://arxiv.org/abs/1903.11816">arXiv:1903.11816, 2019(v1)</a></i>

[2] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. Generative Adversarial Networks. <i>arXiv preprint  <a href="https://arxiv.org/abs/1406.2661">arXiv:1406.2661, 2014</a></i>

[3] Jonathan Long, Evan Shelhamer, Trevor Darrell. Fully Convolutional Networks for Semantic Segmentation. <i>arXiv preprint <a href="https://arxiv.org/abs/1411.4038">arXiv:1411.4038, 2015</a></i>

[4] Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. <i>arXiv preprint <a href="https://arxiv.org/abs/1706.05587">arXiv:1706.05587, 2017 (v3)</a></i>

[5] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. <i>arXiv preprint <a href="https://arxiv.org/abs/1606.00915">arXiv:1606.00915, 2017 (v2)</a></i>

[6] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. Spectral Normalization for Generative Adversarial Networks. <i>arXiv preprint <a href="https://arxiv.org/abs/1802.05957">arXiv:1802.05957, 2018</a></i>

[7] Wei-Chih Hung, Yi-Hsuan Tsai, Yan-Ting Liou, Yen-Yu Lin, Ming-Hsuan Yang. Adversarial Learning for Semi-Supervised Semantic Segmentation. <i>arXiv preprint <a href="https://arxiv.org/abs/1802.07934">arXiv:1802.07934, 2018 (v2)</a></i>

[8] Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. <i>arXiv preprint <a href="https://arxiv.org/abs/1609.05158">arXiv:1609.05158, 2016</a></i>
