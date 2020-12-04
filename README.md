# Vehicle Type Recognition from image data

This project aims to recognize vehicles through images. It is based on a Convolutional Neural Network (from now named CNN) following the essence of Machine Learning algorithms.

The intention is to classify a dataset composed of images that includes different types of vehicles including cars, bicycles, boats, trucks, etc. with a total of 15 categories.

The dataset has been collected from the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html) (over 9 million images) using a subset, selected to contain only vehicle categories among the total of 600 object classes.

The data contains a folder of training data with the class labels and a folder of test data without the labels. The model deals to predict the secret labels for the test data. So, this is an image classification task.

![Image](https://i.gzn.jp/img/2017/08/08/what-is-deep-learning/a16.jpg)

---

## **Details of dataset available**
![Image](https://i.imgur.com/4ZHN8kk.png)

The dataset consists of two files listed below.

- The training set (train.zip): a set of images with true labels in the folder names. The zip file contains altogether 27290 files organized in folders. The folder name is the true class; i.e., "Boat" folder has all boat images, "Car" folder has all the car images and so on.
- The test set (test.zip): a set of images without labels. The zip file contains altogether 7958 files in a single folder. The file name is the id for the solution's first column; i.e., the predicted class for file "000000.jpg".


## **First step: Exploratory Data Analysis**
![Image](https://media.giphy.com/media/l378c04F2fjeZ7vH2/giphy.gif)

In this first phase, I analyze the data that I have available for training the neural network. For this I ask ourselves the following questions that have been resolved in the corresponding notebooks:

- How many images do we have to train?
- How many images do we have to test?
- How many images do we have per class?
- Is there a very unbalanced class? (If a class has many more samples than another class, the data set is not balanced.)
- In order to visualize the dataset, I use the matplotlib library to display a random subset of images for the different classes.

In addition, I cleaned the training dataset. It can be seen that image classes may contain peculiar or odd images. Therefore, to enhance the dataset quality, I filtered the data and removed these "outliers" from the training set.

## **Second step: Train a predictive model**
![Image](https://1.bp.blogspot.com/-xesSQJEhiGI/Vmu62yglkfI/AAAAAAAAFhE/TZRzVdMg-xM/s1600/GPR.png)
I started the project and learning process by building my own CNN models. Testing how adding different layers affect on validation loss. 
Validation loss was my primary score during training. Own models were prone to overfitting, but batch normalization after each convolution helped. 
Also, I did not use yet image augmentation with my own models. Could not reach 80% accuracy and kept in mind that I can not create competitive CNN model without deep knowledge so 
I switched to pretrained models offered by Keras.

### Model parameters
First trained a simple model sequential to achieve fast training and quickly test how model parameters like pooling affects. Testing also that how many Dense layers should be added on top of the model.

After that, I tested pre-trained models with keras and added the sequential model as base model.

### Data augmentation and training
To load images, I used Keras Image Generators, which generate batches of tensor image data with real-time data augmentation. In particular, I used horizontal flip, rotation, width shift, and height shift. In this setup, zoom augmentation did not give any gain in the evaluation score.

If training accuracy tend to go higher than validation accuracy (or loss lower) I added more augmented image sets and decreased epochs for each set.

I used 85% of images for training and 15% for validation. Stratified and different random_state for each model.

After the val loss did not decrease anymore I finalized the model by training quickly with validation images. I left the original unprocessed images for validation and run the same sequential augmentation and training for the previous validation images.

### Models trained
It can be noticed, that certain classes are dominant in the data set (big class imbalance). My solution was to use sklearn.utils.class_weight as class weights when training networks.

I tried the following architectures: Sequential, InceptionV3, MobileNet, ResNet50, NasNetLarge, EfficientNet. Different image sizes starting from 224 and up tu 331 as well as batch sizes were used.

MobileNet and ResNet50 seemed to be the best.

> Here you have a tutorial [MobileNet](https://keras.io/api/applications/mobilenet/) and [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function).


## **Streamlit as visualization result**
![Image](https://mark.douthwaite.io/content/images/2020/09/image-1.png)
It is a library that makes it easy to create web applications to display results of your data analysis.

### :computer: **Technology stack**
- Python
- numpy
- pandas
- matplotlib
- seaborn
- Image
- cv2
- keras
- tensorflow
- sklearn

All processes are built on a data pipeline through PyCharm IDE. Only need to run main_script.py

### :boom: **Core technical concepts and inspiration**
This project was born from the need to apply the knowledge learned during the data analytics bootcamp in a real application in my daily work.

### :shit: **ToDo**
Possible new steps into the project:
- Road speed according to each predicted vehicle.
![Image](https://www.researchgate.net/profile/Fabien_Moutarde2/publication/45876940/figure/fig2/AS:306023719030808@1449973148994/Speed-limit-sign-detection-on-US-road-case-of-most-common-type-of-sign.png)
- Vehicles, pedestrians and sign lights recognition
![Image](https://ak.picdn.net/shutterstock/videos/33521569/thumb/10.jpg)

---
