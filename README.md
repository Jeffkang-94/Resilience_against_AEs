# Malware Image Classfication

We use `Microsoft Malware Classification Challenge (BIG2015)` dataset.
To classify the malware examples, we utilize the image classification task using DNN models, e.g., InceptionV3.
Given the binary samples, we can convert them into the images based on the Incremental Coordinate(IC) Method.
Readers may refer to this paper to craft the malware images.
Malware image samples are represented as below:
<img src="https://github.com/Jeffkang-94/Resilience_against_AEs/blob/main/asset/malware%20samples.png" width="800" height="400">
## Preprocessing
```
cd classification
python split.py # spliting the train and test data
```
## Dataset
Family name | Number of samples | Type
----- | ----- | -----
Ramnit| 1,541| Worm
Lolipop | 2,478 | Adware
Kelihos ver3| 2,942 |Backdoor
Vundo |475 |Trojan
Simda |42 |Backdoor
Tracur |751 |Trojan Downloader
Kelihos ver1 |398 |Backdoor
Obfuscator.ACY |1,228 |Any kind of obfuscated malware
Gatak |1,013 |Backdoor
Sum |10,868|
## Training the Malware Classifier
```
> CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size 100 \
    --lr 0.01 \
    --name checkpoint

- `batch_size` is a size of batch.
- `lr` is a learning rate to update various losses.
- `name` is a name of folder. checkpoint will be stored on this folder.


## Generate AEs with additive gaussian white noise
You can easily generate the noisy samples using below code.
```
python make_noise.py
```
Note that, we set the `stddev` as small enough value since generated samples are supposed to be indistinguishable with the original samples.
We empirically found that `0.001` does not hurt structural features yet effective enough to deceive the target classifier.
```python
noise = torch.zeros_like(img).cuda()
noise.data.normal_(0, 0.001)
img = noise+img
```

## Training the cDCGAN

Before training the model, we randomly pick the 32 images per class.
```
python pick.py # sampling 32 images per class
python train.py # training the cDCGAN model
```

<img src="https://github.com/Jeffkang-94/Resilience_against_AEs/blob/main/asset/cDCGAN.png" width="800" height="400">

