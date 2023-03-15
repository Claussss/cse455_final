For this competition, we are required to solve an image classification task to identify bird species.


# Dataset

There are **555** classes, approximately **38,000** images of varying sizes in the **training** set, and **10,000** in the **testing** set.

Below is some sample data.

![image](https://user-images.githubusercontent.com/48916506/224579418-43ff71a6-7d2a-4b1e-9870-b79a3f4594df.png)

![image](https://user-images.githubusercontent.com/48916506/224579443-bd865454-0a9f-488b-be52-f15e0e3a1bc0.png)

A portion of the distribution of the training dataset is presented here.

![image](https://user-images.githubusercontent.com/48916506/224589495-1f0c41b0-a5ee-4b02-a66f-07e0b8f70d9a.png)

It can be observed that the dataset is not evenly distributed. Some classes have as few as 10 samples, while others have as many as 100 samples. On average, each class has approximately 70 samples.

# Balancing the data

We have explored two ways to balance the number of instances for each class.

1) Offline Augmentation. It has been conducted by randomly sampling images with replacement for each class until 100 images per class were obtained. Random rotation, x/y-axis translation, horizontal flipping, and blurring were used for augmenting the sampled images. Offline augmentation may be more effective than that performed during training as the transformations are applied to each image, as opposed to a batch, thus introducing greater variability.

![image](https://user-images.githubusercontent.com/48916506/225381811-e058162b-890b-4347-a3df-a32012b6dbeb.png)

![image](https://user-images.githubusercontent.com/48916506/225382050-f56dfa03-43a1-47c5-bfa3-421a7809e35a.png)

2) Class weights. It allows the loss function to penalise the model more severely for making mistakes on the classes that are inadequately represented. The weights were computed for every class by subtracting the frequency of that class from 1.

Unfortunately, the offline augmentation performed 1% worse on the final testing in comparison to the class weights approach, despite the fact that the min and max class weights differed only in the third decimal point. We believe that offline augmentation may outperform class weights when another set of transforms is chosen. However, for the remainder of this blogpost, we will adhere to class weights balancing.


# Data Augmentation
Data augmentation is a technique for artificially increasing the number of samples in the training dataset in order to prevent the model from overfitting. Transformations that are applied should maintain the data's coherence with the distribution from which it was drawn, as extreme parameters may impede the model's ability to learn.

**In order to optimize the image loading process, we have resized all the images to 256x256 and stored them on an SSD as the new dataset.**

As our final train transforms, we have stopped on resizing images, randomly cropping them, applying a random horizontal flip and utilizing ImageNet normalization. For the test, we only used resizing and ImagetNet normalization.


![image](https://user-images.githubusercontent.com/48916506/224594398-21c4122b-c97f-47f3-b6b3-6a9722cd7868.png)



We have attempted to utilize other transformations such as **random vertical flips** and **rotations**, however, this did not yield an improvement in performance. As there are virtually no images of birds that are inverted in the training dataset, **color jittering** was also ineffective, as color is a crucial feature for classifying a bird's species. By changing the color, it could potentially change the class, making it substantially more complicated for the model. **RandomErasing** was also unsuccessful as the sizes of birds in the images vary greatly, thus often erasing the entirety of the bird, preventing the model from classifying it. 

We have also attempted offline augmentation to increase the number of images, doubling the initial amount (~80K); however, this did not result in a significant improvement in performance, relative to training on the original ~38K images. As such, we have decided to remain with the ~38K images for the duration of this blog post.

# Training

We make use of **PyTorch** as our principal ML framework. The only components taken from preexisting work are the data_processing function and the training loop, which were taken from the following [tutorial](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing).

## Model
We have been experimenting with various versions of ResNet, such as Resnet18, ResNet101, and ResNet152. However, Resnet18 was deemed too small to learn complex features, while ResNet101 and 152 were too large, leading to overfitting. Ultimately, ResNet50 was chosen as the ideal balance between generalization and data learning. We attempted to add a self-attention mechanism to the ResNet50 model, but it did not yield better performance, thus the attention block was omitted to reduce training time. Variations of ResNet50 such as ResNetX50, which implements certain optimizations, were trialled, yet the performance was not better than that of the pure ResNet50. This is because the quality of the pretrained weights on ImageNet for ResNetX50 was significantly worse than for the pure version. As a result, for the remainder of this blog post, the pure ResNet50 pretrained on ImageNet will be utilized.


## Hyperparameters
In order to monitor the performance of the model, we have created a **validation dataset** comprising **20%** of the training dataset. As the principal performance metric, we utilize **accuracy** and CrossEntropy as the loss function. As the optimizer, we employ **SGD** with **custom weight decay and momentum**. We have tried different batch sizes, and it didn't influence the performance much although smaller batch sizes tend to help to prevent overfitting. Thus, we used **batch size 64** to maximum load our GPU and prevent the model from overfitting.

## Progressive resizing
Progressive resizing is a technique to improve the performance of a model by gradually increasing the resolution of the input images during training.


Initially, we trained the model on 128x128 images for 5 epochs with a learning rate of 0.01. Following this, we will continue to train the same model on 224x224 images for 30 epochs, with the learning rate decreasing to 0.001 at the 10th epoch and 0.0001 at the 17th epoch. The weight decay and momentum will remain stable at 0.0005 and 0.9, respectively.

We have experimented with various Pytorch schedulers, however, the manually adjusted learning rate produced the most desirable results.

# Evaluation
Prior to evaluating the model on the test dataset, we have retrained it on the entirety of the dataset without dividing into a validation set.

## TTA
TTA stands for Test Time Augmentation, which is a technique used in machine learning to improve the accuracy of a model's predictions. It involves applying data augmentation techniques to the test data during the inference stage, which can help to reduce overfitting and improve the model's ability to generalize to new data.
In our implementation, we apply the same transforms we used for training to every test image four times, and make a prediction for each augmented image. We then average the final class probabilities, resulting in a more stable prediction. This yields an approximate 2% increase in accuracy.


The final pipeline achieved a score of 83% on the test set.

## Future Work
We think further work on the dataset might give some performance boost. For example, instead of just using class_weights (which are very close to each other because the number of samples is huge), we could add some samples generated with Mixup augmentation or Conditional GANs. Additionally, given a larger dataset, Vision Transformers have shown to perform better than classic CNNs, and as such, it may be advantageous to try.


[Main Training Pipeline](https://www.kaggle.com/code/yuriihalychanskyi/cse-455-final-birds)


[Offline Augmentation Script](https://github.com/Claussss/cse455_final/blob/main/offline_augmentation.py)

Team Members:

Yurii Halychanskyi (yhalyc)

Tyler Schwitters (tschwi)
