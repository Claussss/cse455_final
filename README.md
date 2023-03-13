For this competition, we are required to solve an image classification task to identify bird species.


# Dataset

There are **555** classes, approximately **38,000** images of varying sizes in the **training** set, and **10,000** in the **testing** set.

Below is some sample data.

![image](https://user-images.githubusercontent.com/48916506/224579418-43ff71a6-7d2a-4b1e-9870-b79a3f4594df.png)

![image](https://user-images.githubusercontent.com/48916506/224579443-bd865454-0a9f-488b-be52-f15e0e3a1bc0.png)

A portion of the distribution of the training dataset is presented here.

![image](https://user-images.githubusercontent.com/48916506/224589495-1f0c41b0-a5ee-4b02-a66f-07e0b8f70d9a.png)

It can be observed that the dataset is not evenly distributed. Some classes have as few as 10 samples, while others have as many as 100 samples. On average, each class has approximately 70 samples.

# Data Augmentation
Data augmentation is a technique for artificially increasing the number of samples in the training dataset in order to prevent the model from overfitting. Transformations that are applied should maintain the data's coherence with the distribution from which it was drawn, as extreme parameters may impede the model's ability to learn.

**In order to optimize the image loading process, we have resized all the images to 256x256 and stored them on an SSD as the new dataset.**

As our final train transforms, we have stopped on resizing images, randomly cropping them, applying a random horizontal flip and utilizing ImageNet normalization. For the test, we only used resizing and ImagetNet normalization.


![image](https://user-images.githubusercontent.com/48916506/224594398-21c4122b-c97f-47f3-b6b3-6a9722cd7868.png)



We have attempted to utilize other transformations such as **random vertical flips** and **rotations**, however, this did not yield an improvement in performance. As there are virtually no images of birds that are inverted in the training dataset, **color jittering** was also ineffective, as color is a crucial feature for classifying a bird's species. By changing the color, it could potentially change the class, making it substantially more complicated for the model. **RandomErasing** was also unsuccessful as the sizes of birds in the images vary greatly, thus often erasing the entirety of the bird, preventing the model from classifying it.

# Training

We make use of **PyTorch** as our principal ML framework. The only components taken from preexisting work are the data_processing function and the training loop, which were taken from the following [tutorial](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing). 

## Class Weights
Prior to beginning training, we have attempted to address the issue of imbalanced classes by calculating **class weights** by computing (1-class_frequency) for every class and passing it to the **Cross Entropy loss** function, which is employed in our case. Thus, when calculating gradients, the model will be penalised more severely for making mistakes on the classes that are inadequately represented.

## Hyperparameters
In order to monitor the performance of the model, we have created a **validation dataset** comprising **20%** of the training dataset. As the principal performance metric, we utilize **accuracy**. As the optimizer, we employ **SGD** with **custom weight decay and momentum**.We have tried different batch sizes, and it didn't influence the performance much although smaller batch sizes tend to help to prevent overfitting. Thus, we used **batch size 128** to maximum load our GPU.

## Progressive resizing
In order to reduce overfitting, we will train a pretrained ResNet50 model on 128x128 images for 5 epochs with a learning rate of 0.01. Following this, we will continue to train the same model on 224x224 images for 30 epochs, with the learning rate decreasing to 0.001 at the 10th epoch and 0.0001 at the 17th epoch. The weight decay and momentum will remain stable at 0.0005 and 0.9, respectively.

We have experimented with various Pytorch schedulers, however, the manually adjusted learning rate produced the most desirable results.

# Evaluation
Prior to evaluating the model on the test dataset, we have retrained it on the entirety of the dataset without dividing into a validation set.
## TTA
TTA stands for Test Time Augmentation, which is a technique used in machine learning to improve the accuracy of a model's predictions. It involves applying data augmentation techniques to the test data during the inference stage, which can help to reduce overfitting and improve the model's ability to generalize to new data.
In our implementation, we apply the same transforms we used for training to every test image four times, and make a prediction for each augmented image. We then average the final class probabilities, resulting in a more stable prediction. This yields an approximate 2% increase in accuracy.


The final pipeline achieved a score of 83% on the test set.

## Future Work
We think further work on the dataset might give some performance boost. For example, instead of just using class_weights (which are very close to each other because the number of samples is huge), we could add some samples generated with Mixup augmentation or GANs. Once we have more images, we can try applying bigger models like ResNet101, some Vits.


