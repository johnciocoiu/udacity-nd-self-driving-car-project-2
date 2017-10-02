# Traffic Sign Recognition

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set) *[CHECK]*
* Explore, summarize and visualize the data set *[CHECK]*
* Design, train and test a model architecture *[CHECK]*
* Use the model to make predictions on new images *[CHECK]*
* Analyze the softmax probabilities of the new images *[CHECK]*
* Summarize the results with a written report *[CHECK]*

[//]: # (Image References)

[visualization]: ./img/visualization.png "Visualization"
[customimg1]: ./img/test/1.png "Custom image 1"
[customimg2]: ./img/test/2.png "Custom image 2"
[customimg3]: ./img/test/3.jpg "Custom image 3"
[customimg4]: ./img/test/4.jpg "Custom image 4"
[customimg5]: ./img/test/5.jpg "Custom image 5"
[customimg1crop]: ./img/1.png "Custom image cropped 1"
[customimg2crop]: ./img/2.png "Custom image cropped 2"
[customimg3crop]: ./img/3.png "Custom image cropped 3"
[customimg4crop]: ./img/4.png "Custom image cropped 4"
[customimg5crop]: ./img/5.png "Custom image cropped 5"
[preprocess]: ./img/preprocess.png "Pre-processed image"

## Step by step summary of the code

In this writeup I give extra information and supports the [HTML file](https://github.com/johnciocoiu/udacity-nd-self-driving-car-project-2/blob/master/Traffic_Sign_Classifier.html) I made of for this traffic sign recognition project.

---

### Step 1. Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3), 32x32 pixels over 3 color channels.
* The number of unique classes/labels in the data set is 43

#### Visualization of the set
![visualization]

The visualization shows:
* Distribution over the signs
* Similar shape of distribution over test-, train-, and validation set

### Step 2. Design and Test a Model Architecture

#### Pre-processing

The following pre-processing steps were performed:
* Second step was normalizing the data, because the images will get a similar zero mean and equal variance. This gives the network less noise.
* I did NOT take the grayscale, as in fact my network performed worse with it. Also, in [this paper](http://people.idsia.ch/~juergen/ijcnn2011.pdf) (see below for more info), is said that day have the same.

![alt text][preprocess]

#### Used model

My final model consisted of the following layers:

| Layer         		|     Description	        					| Output    | 
|:---------------------:|:---------------------------------------------:|:---------:| 
| Input         		| Image            					    		| 32x32x3   |
| Convolution 3x3     	| 1x1 stride, same padding                  	| 30x30x6   |
| RELU					| Activation    								| idem      |
| Max pooling	      	| 2x2 stride                     				| 15x15x6   |
| Convolution 3x3     	| 1x1 stride, same padding                  	| 13x13x16  |
| RELU					| Activation									| idem      |
| Max pooling	      	| 2x2 stride                     				| 6x6x16    |
| Convolution 3x3     	| 1x1 stride, same padding                   	| 4x4x32    |
| RELU					| Activation									| idem      |
| Flatten               | Flatten previous features                     | 512       |
| Fully connected		| Non-linear function       					| 160       |
| Dropout               | Remove noise                                  | idem      |
| RELU					| Activation									| idem      |   
| Fully connected		| Non-linear function       					| 100       |
| Dropout               | Remove noise                                  | idem      |
| RELU					| Activation									| idem      |
| Fully connected		| Non-linear function       					| 43        |

#### Model training

To train the model, I used my own computer. My company has a VM with a great GPU, but my laptop was fast enough to let me do these calculations on it. With some trial and error, I came with the following variables used for training:

* The used batch size is 1024, that makes it a little faster on my laptop
* I used 30 epochs, because I saw the accuracy almost settles the same from that point
* The learning rate used is 0.005, makes it faster than the given 0.001
* The optimizer used was a 'adam'-optimizer, for minimalizing the loss function (cross entropy)
* For max-pooling, I used a keep probability of 0.7

#### Validation

To get a better validation accuracy, I've read the [given paper [1]](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), but I also used [this paper [2]](http://people.idsia.ch/~juergen/ijcnn2011.pdf), which is the paper of the best network according to accuracy given in the first paper. What I did to get the 0.93 validation:

* First I tried it with the given LeNet-approach.
* I added a third Convolution layer, like they also did in paper [2]. 
* Changed the shapes of the convolution layers, like they also did in the paper [2]. But, there they had images of 48x48, that's why I adjusted the shapes of the layers to fit on the 32x32 images.
* Added dropouts for reducing the noise and prevent overfitting. I tried with different dropout probabilities. The 0.5 as mentioned the 'most optimal' probability, scored worse than 0.7.
* Tried with and without grayscaling the image. My model scored better without grayscaling.

My final model results were:
* training set accuracy of *0.990*
* validation set accuracy of *0.927*
* test set accuracy of *0.915*

This model is working fine, because the accuracy on the test data (which I never used before) is now above the 90%, which is good enough. Although, when designing a real 'self-driving car', the accuracy has to be higher to reduce the chance of having an accident.
 
### Step 3. Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][customimg1] 
![alt text][customimg2] 
![alt text][customimg3] 
![alt text][customimg4] 
![alt text][customimg5]

I cropped every image at input loading to the following images:

![alt text][customimg1crop] 
![alt text][customimg2crop] 
![alt text][customimg3crop] 
![alt text][customimg4crop] 
![alt text][customimg5crop]

#### Prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)	| Turn right ahead         						| 
| Stop     			    | Stop 									    	|
| Yield					| Yield											|
| Roundabout mandatory	| Roundabout mandatory							|
| Ahead only			| Ahead only      						    	|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the accuracy on the test set of *0.915*. 

#### Soft-max

The code for making predictions on my final model is located in the HTML file.

These are the probabilities for the top-5 sign-types per image:

##### Image 1:

| Probability | Sign |
|:-----------:|:----:|
| 1.000 | 35 (Ahead only) |
| 0.000 | 0 (Speed limit (20km/h)) |
| 0.000 | 3 (Speed limit (60km/h)) |
| 0.000 | 39 (Keep left) |
| 0.000 | 38 (Keep right) |

##### Image 2:

| Probability | Sign |
|:-----------:|:----:|
| 1.000 | 14 (Stop) |
| 0.000 | 3 (Speed limit (60km/h)) |
| 0.000 | 5 (Speed limit (80km/h)) |
| 0.000 | 29 (Bicycles crossing) |
| 0.000 | 17 (No entry) |

##### Image 3:

| Probability | Sign |
|:-----------:|:----:|
| 1.000 | 13 (Yield) |
| 0.000 | 9 (No passing) |
| 0.000 | 10 (No passing for vehicles over 3.5 metric tons) |
| 0.000 | 39 (Keep left) |
| 0.000 | 12 (Priority road) |

##### Image 4:

| Probability | Sign |
|:-----------:|:----:|
| 1.000 | 40 (Roundabout mandatory) |
| 0.000 | 6 (End of speed limit (80km/h)) |
| 0.000 | 38 (Keep right) |
| 0.000 | 32 (End of all speed and passing limits) |
| 0.000 | 42 (End of no passing by vehicles over 3.5 metric tons) |

##### Image 5:

| Probability | Sign |
|:-----------:|:----:|
| 1.000 | 35 (Ahead only) |
| 0.000 | 37 (Go straight or left) |
| 0.000 | 36 (Go straight or right) |
| 0.000 | 33 (Turn right ahead) |
| 0.000 | 34 (Turn left ahead) |

