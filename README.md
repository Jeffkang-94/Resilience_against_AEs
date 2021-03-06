# Malware Image Classfication

This repository covers the malware image classification and malware image generation.
We use `Microsoft Malware Classification Challenge (BIG2015)` dataset.
To classify the malware examples, we utilize the image classification task using DNN models, e.g., InceptionV3.
Given the binary samples, we can convert them into the images based on the Incremental Coordinate(IC) Method.
Readers may refer to this **[paper](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201823952425926&dbt=NART)** to generate the malware images.
Malware image samples generated by IC method are prsented as below. 
We depict two samples per class from 1 to 9.

<img src="https://github.com/Jeffkang-94/Resilience_against_AEs/blob/main/asset/malware%20samples.png" width="800" height="400">


## Preprocessing

### Dataset
`Microsoft Malware Classification Challenge (BIG2015)`
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


### How to convert binary to images
To apply DNN-based malware classification, each malware sample should be converted into
a corresponding image of fixed size.  
This figure illustrates how malware files are converted into Incremental Coordinate image files.
First, the malware binary is read in 4-byte increments. The first 2 bytes are converted to x coordinates, the last 2 bytes are converted to y coordinates, the value represented by those coordinates is incremented by 1. This process is performed until the end of the file is reached. This generates a 256 by 256 gray scale image from each malware sample. The key advantage of visualizing a malicious PE file is that it can perform malware detection/classification without running malware. In addition, small changes that were not detectable in malware binaries may be detectable in malware converted to images


<img src="https://github.com/Jeffkang-94/Resilience_against_AEs/blob/main/asset/convert.png" width="800" height="300">

### Split the dataset
We divided the dataset into 9:1 ratios using the following code.

```
cd classification
python split.py # spliting the train and test data
```

## Training the Malware Classifier
```
> CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size 100 \
    --lr 0.01 \
    --name checkpoint
```
- `batch_size` is a size of batch.
- `lr` is a learning rate to update various losses.
- `name` is a name of folder. checkpoint will be stored on this folder.


### Generate AEs with additive gaussian white noise
You can easily generate the noisy samples using below code.
```
python make_noise.py
```
Note that, we set the `stddev` as small enough value since generated samples are supposed to be indistinguishable with the original samples.
We empirically found that `0.001` does not hurt structural features yet effective enough to deceive the target classifier.
Additive noise will incur the error amplification effects through the intermediate layer of the classifier, resulting in misclassification.
```python
noise = torch.zeros_like(img).cuda()
noise.data.normal_(0, 0.001)
img = noise+img
```

### Evaluate the noise data with pre-trained classifier
Noise inputs will be stored at `noise` folder, and you can evaluate the pre-trained classifier with the noise inputs.
Detail of the experimental results can be found on the original paper, *Resilience against Adversarial Examples:
Data-Augmentation Exploiting Generative Adversarial Networks, TIIS, 2021*
```
python test.py # output the accuracy
```



## Training cDCGAN

Before training the model, we randomly pick the 32 images per class.
```
python pick.py # sampling 32 images per class
python train.py # training the cDCGAN model
```

<img src="https://github.com/Jeffkang-94/Resilience_against_AEs/blob/main/asset/cDCGAN.png" width="800" height="400">


## Re-Training the classifier

Once cDCGAN is fully trained, we can leverage the model to generate extra data to enhance the baseline model.
In the simple experiment, we adjust the number of samples and evaluate the performance based on the classification accuracy.


### Evaluation 
Metric/Test | Baseline | n=10 | n=20 | n=100
----- | ----- | ----- | ----- | -----
original(balanced) | 97.4(96.4) | 97.4(95.4) | 97.2(95.2) | 97.5(95.4)
AEs(balanced) | 71.1(59.2) | 74.56(66.78) | **75.11(70.16)** | 67.3(65.41)
Average FPR | 3.78  | 2.97 | 3.0 | 3.8


### ROC curve
<img src="https://github.com/Jeffkang-94/Resilience_against_AEs/blob/main/asset/ROC%20curve.png" width="800" height="400">
