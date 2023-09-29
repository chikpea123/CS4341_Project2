# CS4341_Project2
Image classification using machine learning

## 1. Project Description & Dataset 

### Primary Goal: 

In this project, you will learn how to build a Deep Neural Network for image classification and quantitatively evaluate the network. The project will include three components: (1) data processing (2) model architecture and hyperparameter selection (3) evaluation and visualization. You will write a program that takes an image of a flower as input and outputs the name of the flower present in the image. 

### Dataset: 

The Tensorflow flower dataset is a 5-category flower image dataset with roughly 800 images for each class. Each color image usually has three channels: red, green, and blue (RGB image). All images have an associated ground truth annotation of flower name. The dataset is roughly 200 MB in size.  Its homepage is https://www.tensorflow.org/tutorials/load_data/images, and the images can be downloaded from the link: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

You are also encouraged to explore the dataset using the image visualization tools on the website: https://knowyourdata-tfds.withgoogle.com/#dataset=tf_flowers

### Tasks: 

In this project, you will experiment with training machine learning models for identifying which flower is present in an image. You will then submit a report describing your experimental methods and results. You will use a package called Keras
Links to an external site. implemented in Python3. Keras is an API for TensorFlow. The input to your model will be an image, and the output will be a classification of the flower in the image, from 0 to 4. You’ll get to work with common machine learning packages used in state-of-the-art research. In addition, you will see how the numerical computation package Numpy is used for preprocessing. 

## 2. Project Requirements 

Your project submission must include a written report and code for training and visualization, which you need to write inside the code template provided. Details of what to include are listed below: 

### (1) Written Report
### Model & Training Procedure Description 
Include a section describing the set of experiments that you performed as follows: 
1. Data preprocessing implementation (i.e, shuffle or rotate data); 
2. ANN architectures (i.e, number of layers, number of neurons in each layer, type of activation function); 
3. Hyper-parameter value selection (i.e, the reshaped input image size, the type of optimizer, training epochs, and number of neurons or hidden_size or number of parameters, the loss function); 
4. Accuracy you obtained in each of these experiments on the test set. 

### Model Performance & Confusion Matrix 

Include a section describing in more detail the most accurate model you were able to obtain: the ANN architecture of your model, including number of layers, number of neurons in each layer, activation function, number of epochs used for training, type of optimizer used for training, type of loss function used for training, values for other hyperparameters, and data preprocessing used. 

Include a confusion matrix, showing the results of testing the model on the test set. The matrix should be a 5-by-5 grid showing which categories of images were classified. Use your confusion matrix to additionally report precision and recall for each of the 5 classes, as well as the overall accuracy of your model.  

### Training Performance Plot 

For your best-performing ANN, include a plot showing how training accuracy and validation accuracy change over time during training. Graph number of training epochs (x-axis) versus training set and validation set accuracy (y-axis). Hence, your plot should contain two curves.  


### Visualization 

Include 3 visualizations of images that were misclassified by your best-performing model and any observations about why you think these images were misclassified by your model.  You will have to create or use a visualization program that takes a 30-by-30 matrix input and translate it into a black-and-white image. 

### (2) Code: 

### Model Code 

Please turn in your preprocessing, model creation, model training and plotting code. 
The provided code template itself is complete and executable. The default classification accuracy for this project is around 0.2 which is close to a random guess.   
** Important: You need to edit only the sections of the code surrounded by the "MAGIC HAPPENS HERE" comments. **

### (3) Model: 

### Copy of Best Performing Model: 

Turn in a copy of your best model renamed as `best_trained_model.<ext>' . You need to do “whole-model saving (configuration + weights)” using TensorFlow SavedModel format
Links to an external site..  Please see more information at Keras’ methods for saving your model Links to an external site.. The code template has already included this. 

## 3. Project Preparatory Tasks and Guidelines 

### (1) Installing Software and Dependencies

The code template is written with the Keras API in a Python3 script. You will use this template to build and train a model. To do so, you will need to implement the project in Python3 and install Keras and its dependencies. Please make sure you have a working version of Python3 and Keras as soon as possible, as these programs are necessary for completing the project. For questions about Keras, you can check their FAQ page. 

You can run this command to install the required python package (requirements.txt)

Download python package (requirements.txt).

pip install -r requirements.txt

### (2) Preprocessing Data

The dataset can be downloaded from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

Links to an external site. , and should be unzipped in the folder images. The raw image files should be in the path ./images/flower_photos.

The template already has the code snippets for data loading. You can download the template python code file (provided at the beginning of this project description) and edit it. 

You can convert the loaded tensorflow dataset into numpy array using 

```
images = np.concatenate([x for x, y in train_ds], axis=0)
labels = np.concatenate([y for x, y in train_ds], axis=0)
```

Image data is provided as 30-by-30 matrices of integer pixel values from the range 0-255. Note that in the code template we have already flattened the data for you as ```x = layers.Flatten()(x)``` (This will convert the 30-by-30 2D matrix into 30*30=900 1D vector.)

and scale the values to the range 0-1 as ```x = layers.Rescaling(1./255)(inputs)```

### (3) Setting Hyperparameters

image.png

Effective tuning of hyperparameters is essential for training a neural network model that performs well. In this project, you will work with four crucial hyperparameters:

    Batch Size: The batch size determines the number of data instances used in each iteration of training. Smaller batch sizes may lead to slower convergence but less memory consumption, while larger batch sizes can speed up training but might require more memory.

    Image Size: This hyperparameter defines the dimensions of your input images be resized into. It is important to choose an image size that matches the model architecture. You can try different image sizes like 16*16, 30*30, 50*50, 80*80 and so on.

    Epochs: The number of training epochs controls how many times your model will iterate over the training dataset. 

    Optimizer and Learning Rate: The choice of optimizer and its associated learning rate plays a critical role in training. Optimizers adjust the model's weights during training, and the learning rate determines the step size for these adjustments. Finding the right optimizer and learning rate is often an iterative process. You can try different types of optimizers like sgd, adam, rmsprop and so on.

Properly configuring these hyperparameters is a crucial step in building an effective neural network model. It involves experimentation and understanding how each hyperparameter choice impacts the model's performance.

### (4) Building a model

image.png

In Keras, Models are defined as a composition of layers, you can find the available layers API in Keras Layers

Links to an external site.; you must use this code template as a starting point for building your model. The template includes a sample first input layer and output layer. This portion of the project will involve experimentation. 

Only edit the sections of the code between the "MAGIC HAPPENS HERE" and "MAGIC ENDS HERE" comments and leave the final layer as it appears in the template.

### (5) Compiling and Training a model

image.png

Prior to training a model, you must specify what your loss function for the model is and what your gradient descent method is in the config variable.

You have the option of changing how many epochs to train your model for and the types of optimizer in the config variable. Experiment to see what works best. Also, remember to include the validation data in the fit() method.

### (6) Reporting Your Results

fit() returns information about your training experiment. In the template, this is stored in the “history” variable. Use this information to construct your graph that shows how validation and training accuracy change after every epoch of training.

Use the predict() method on the model to evaluate what labels your model predicts on the test set. Use these and the true labels to construct your confusion matrix, like the toy example below, although you do not need to create a fancy visualization. Your confusion matrix should have 5 rows and 5 columns (one for each of the class labels). You can use function confusion_matrix from package sklearn. For the plotting, please check the function heatmap from package matplotlib. In addition to the confusion_matrix, you have to report the precision and recall for each of the 5 classes.

image.png

Use matplotlib to generate the image of the misclassified samples.