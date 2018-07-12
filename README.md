# Udacity_SDC_NanoDegree_Term1_Project2_Xin
## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of cross-validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 43 classes are shown here with the index:

![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/data_set.png)

Then the data set distribution is shown by the bar chart

![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/original_data.png)

The number of images for some classes are far less than enough and therefore data augmentation is necessary before training.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I used the several methods from the skimage library to increase the number of images. Here, four methods are used: 
 * add random noise
 * random rotataion of images between -20 and 20 degrees
 * color inversion
 * image blurring
 * intensinty rescale
 
Here are some exmples of the obtained traffic sign images:

 ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/augmentation.png)
 
 ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/augmentation2.png)
 
 ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/augmentation3.png)
 
 During the data augementation, these methods are chosen randomly on different images and here I increased the data set 3 times. Then I converted the images to grayscale and normalized the image data. In the end, the images are reshuffled with a 85% for training and 15% for validation. 
 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The CNN is based on the architecture of LeNet. Since the problem is more complicated than the problem for LeNet, I decided to added one more convolution layer before the fullly connected layers.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution1 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution2 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution3 5x5     	| 1x1 stride, valid padding, outputs 400 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Fully connected		| Input 420, output 120        									|
| RELU					|												|
| Dropout1		| tuning      									|
| Fully connected		| Input 120, output 84        									|
| RELU					|												|
| Dropout2		| tuning     									|
| Fully connected		| Input 84, output 43       									|
| Softmax				|     									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer and the a batch size of 128 is used to guarentee the training speed.The initial training rate is 0.001 (70 epochs) and then I used the training rate of 0.0001 for fine tuning.The best training result obtained is with a second dropout factor of 0.5 for the second fully connected layer and a first dropout factor of 1. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Accuracy = 0.990
* Validation Accuracy = 0.986
* Test Accuracy = 0.926

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I first tested the LeNet directly but the validation as well as the training accuracy are low.

* What were some problems with the initial architecture?

The parameters in the neural network could not be enough and therefore I added one more convolution layer.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

After training the second architecture, I found that the accuracy is quite good but the accuracy of the validation is much lower compared with the training set. Therefore, I added two dropout layers after the fully-connected-layers to reduce the overfitting. A factor of 0.5 for the second dropout layer is found to be approriate to reduce the overfitting..

* Which parameters were tuned? How were they adjusted and why?

Training rate, dropout factors are tuned. Dropout factors are tuned to change the problem of overfitting or underfitting. The training rate is tuned to approach more to the global minimum.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Adding convolution layer can increaes the number of paramters in the neural network, which in turn reduce the bias of the algorithms. The dropout layer is for regularization, which can be used to reduce the variance of the system.

 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/six_new.png)

The images are labeled according to the .csv file that has been read into DataFrame before. Then I preprocessed the images to the same format of the original data set.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy of tested 6 new signs = 1.000. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In the end, I calculated the top 5 softmax probabilities for the 6 new images. Here are the results:
![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/read_signs.png)

Since the accuracy is 1.000, I chose the first image in the training set to show along with the accuracy. For the first sign 0f 60 kv/cm, the accuracy is 0.996 and several other speed limit signs have some probabilities. This could be due to the low intensity of the figure (shown in the previous figure). The probabilities for other images are quite close or equal to one.



