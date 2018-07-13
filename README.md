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

I tested several architectures and here are the links:
* LeNet [project code_LeNet](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/blob/master/Traffic_Sign_Classifier_LeNet.ipynb)
* LeNet_variation (add one more convoluntional layer) [project code_LeNet_Variation](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/blob/master/LeNet_Variation.ipynb)
* simplified VGG [project code_VGG](https://github.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/blob/master/Traffic_Sign_Classifier_VGG.ipynb)

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

Two probelms can be seen here:
* The number of images for some classes are far less than enough and therefore data augmentation is necessary before training.
* The image distribution is strongly imbalanced. Rebalencing the images in different classes is needed to be taken into account during data augmentation.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### 1> Data Normalizaiton.

##### 2> Data Augmentation.
I used the two methods to increase the number of images. Here, four methods are used: 
 * random rotataion of images between -20 and 20 degrees
 * image blurring
 
Here are some exmples of the obtained traffic sign images:

 ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/augmentation.png)
 
 ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/augmentation2.png)
 
 ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/augmentation3.png)
 
 During the data augementation, these methods are chosen randomly on different images. To resolve the imbalance  of the dataset for 43 classes. The generated image numbers are chosen according to the total number of images for each class. The bar chart below shows the resulted distribution:
  ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/new_distribution.png)
 
 ##### 3> Grayscale.
 I converted the images to grayscale 
 
 ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/grayscale.png)
 
 ##### 3> Histogram Equalization.
 Performing local histogram equalization on the dataset to increase the contrast. 
 ![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/equalization.png)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


##### 1> LeNet.
Leraing_rate = 0.001
Epochs = 20
Batch_Size = 32
Results:
* Training Accuracy = 0.994
* Validation Accuracy = 0.971
* Test Accuracy = 0.944

##### 2> LeNet_Vriation.

One convolutional layer is added before the fully connected layer.

Leraing_rate = 0.001
Epochs = 20
Batch_Size = 32
Results:
* Training Accuracy = 0.999
* Validation Accuracy = 0.959
* Test Accuracy = 0.932

##### 3> VGG_variation. 
[CNN architectures](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf)
[VGG](https://arxiv.org/pdf/1409.1556.pdf)

* simplified VGG 

Leraing_rate = 0.001
Epochs = 15
Batch_Size = 32
Results:
* Training Accuracy = 0.996
* Validation Accuracy = 0.981
* Test Accuracy = 0.956



| Layer         		|    
|:---------------------:|
| Convolution1 3x3     		| 
| Convolution2 3x3     	| 
| Max pooling 1				|		
| Convolution3 3x3        	| 
| Convolution4 3x3      	| 	
| Max pooling	2				|											
| Convolution5 3x3     	| 
| Convolution6 3x3       	| 
| Max pooling	3			|											
| Fully connected	1	| 
|  Fully connected		2				|												
|  Fully connected		3	| 
   							

 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer and the a [small batch size](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network) of 32 is used to increase training speed. The training rate is 0.001 for 20 epochs. Dropout factors are tuned as a hyperpaarameter to reduce the overfitting problem.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started by testing the LeNet without data augmentation and the validation accuracy is less than 0.9. Then I increase the total number of the dataset and tried some variations of the LeNet architecture.However, the accuracy is still not enough and there is certain gap between the training accuracy and the test accuracy. Then I repreform the data augmentation to achieve a relatively evenly distributed dataset and the accuracy increase significantly for different architectures. 

During the training, I added dropout layers for all the architectures as a regularization tool, which can reduce the overfitting. I added two several dropout layers after each fully connected layer. A relatively low value of 0.5 is used on the dropout layers. 

The VGG architecture is much deeper and powerful than LeNet. Its learning capability is thus much higher. Only 3x3 convolution and 2x2 pooling are used throughout the whole network. The depth is therefore very important for the performance of the CNN.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 5 German traffic signs that I found on the web:

![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/new_image.png)

The images are labeled according to the .csv file that has been read into DataFrame before. Then I preprocessed the images to the same format as the original data set.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy of tested 5 new signs = 1.000  for the three tested architectures.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In the end, I calculated the top 5 softmax probabilities for the 6 new images. Here are the results:
![alt text](https://raw.githubusercontent.com/chaidamu519/Udacity_SDC_NanoDegree_Term1_Project2/master/top_five.png)

The top_5 images have always shown similar patterns. For example, the speed limit signs are similar and as a result the top 5 figures are all signs for speed limit. Triangle signs are shown in top 5 for the prediction of sign of children crossing.



