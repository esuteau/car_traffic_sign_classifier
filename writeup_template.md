#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Visualizations/input_example.png "Example of Traffic Sign from the dataset"
[image2]: ./Visualizations/distribution_training.png "Distribution of the Training Set"
[image3]: ./Visualizations/distribution_validation.png "Distribution of the Validation Set"
[image4]: ./Visualizations/distribution_test.png "Distribution of the Test Set"
[image5]: ./Visualizations/web_image_set/modified/traffic_sign_1_edited.jpg "Traffic Sign 1"
[image6]: ./Visualizations/web_image_set/modified/traffic_sign_2_edited.jpg "Traffic Sign 2"
[image7]: ./Visualizations/web_image_set/modified/traffic_sign_3_edited.jpg "Traffic Sign 3"
[image8]: ./Visualizations/web_image_set/modified/traffic_sign_4_edited.jpg "Traffic Sign 4"
[image9]: ./Visualizations/web_image_set/modified/traffic_sign_5_edited.jpg "Traffic Sign 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup

This writeup explains the different steps behind my Traffic Sign Classifier.
The code can be found [here](https://github.com/esuteau/car_traffic_sign_classifier.ipynb)

###Data Set Summary & Exploration

####1. The first hurdle in this project was to find a way to automatically download the German traffic Sign Dataset so that anyone could easily retrain my network. It wasn't really part of the project but I thought it would be really useful for training my network on an AWS instance.
I used tqdm and a code snippet from the deep-learning nano degree to do that. It downloads and unzips the data automatically.
Here is how I did it:
```python
# Download the dataset and extract it
base_path = '/home/carnd/car_traffic_sign_classifier/'
traffic_signs_dataset_folder_path = base_path

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('traffic-signs-data.zip'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Traffic Signs Dataset') as pbar:
        urlretrieve(
            'https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip',
            'traffic-signs-data.zip',
            pbar.hook)

if not isfile('train.p'):
    with ZipFile('traffic-signs-data.zip', 'r') as zip:
        print('Extracting the zipfile to {}'.format(traffic_signs_dataset_folder_path))
        zip.extractall()
        zip.close()
```

I used the shape function to get the size of the different datasets, with the following results:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

The first thing I did was to look at an example from the training set. I was curious what the data looked like.
Here is an example of traffic sign:

![alt text][image1]

So the input images are 32x32x3, nicely cropped right around the sign.

Then one thing of interest was the distribution of the data in the 3 different sets.
Here is the distribution of the training set, normalized to ease the comparison:
![alt text][image2]

So the different classes are not randomly distributed in the dataset, which can be a good thing if it actually represents the probabilities to encounter each sign in Germany, but could be an issue if the cross validation and test have totally different distributions.
A simple comparison between the most common sign and the least common showed that the most common sign had around 11.5 more data samples than the least common.

Now let's look at the validation and test sets.

![alt text][image3]

![alt text][image4]

Graphically, the datasets look very similar to the training set, which is a good thing for the project.

###Design and Test a Model Architecture

####1. Pre-Processing

I experimented with just a few different techniques for pre-processing the data: Conversion to grayscale, normalization and shuffling.

Shuffling the data makes a lot of sense to make sure each batch trained has a collection of the different traffic sign classes and does not only train on one class of signs. this ensures that the gradient will converge faster using the batches.

At first, I thought that converting the image to grayscale would be a bad idea, as loosing the color information could be critical to the classification of the signs. I seems that I was wrong and the network trains much faster with only grayscale images. I still believe that maybe a more complex network architecture would take advantage of the RGB colors for classification and reach a higher accuracy.
I guess that working with grayscale helps the network focus on extracting the most important features of each sign.

In the end, I converted the data to grayscale, normalized and shuffled, and with my final network architecture, I get 95% of accuracy on the validation set

####2. Final model architecture

My final model is close to LetNet, with slightly bigger to convoluation and fully connected layers, which helps since we have more classes to classify than in LeNet (43 vs 10)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale							    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 			     	|
| Fully Connected		| 240 Outputs									|
| RELU					|												|
| Fully Connected		| 120 Outputs									|
| RELU					|												|
| Fully Connected		| 43 Outputs									|

####3. Training of the model

I calculated the loss using the cross entropy, with the softmax function on the output logits to calculate probabilies.
Then I used the Adam optimizer to update the weights of the network.

I played with different sizes of batches, and ended up choosing 128 samples per batch and 20 epochs, which gives a reasonable training time for the network on an AWS instance.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Tuning the different parameters of the network is not an easy task. I started from LeNet's architecture, which was I believe a good starting point.

I refactored different functions to pass all parameters in a dictionary so that I can easily test different architectures without much work. I can also enable/disable the pre-processing steps described above. I had something like this:

```python
epochs = 20
batch_size = 128
rate = 0.001
hyperparams = {}
hyperparams['normalization'] = False
hyperparams['shuffle'] = True
hyperparams['greyscale'] = True
hyperparams['epochs'] = epochs
hyperparams['batch_size'] = batch_size
hyperparams['rate'] = rate
hyperparams['mu'] = 0
hyperparams['sigma'] = 0.1
hyperparams['input_depth'] = X_train.shape[3]
hyperparams['filter_size'] = 5
hyperparams['conv1_depth'] = 32
hyperparams['conv2_depth'] = 64
hyperparams['fc1_out_size'] = 240
hyperparams['fc2_out_size'] = 120
```

I also saved all those parameters and the cross-validation accuracy in a text file, so that I could easily retrace my steps to a better architecture if I needed to. Training a new network would just append a new result line to the file.
The content of this file looked like this:

batch_size | conv1_depth | conv2_depth | epochs | fc1_out_size | fc2_out_size | filter_size | greyscale | input_depth | mu | normalization | rate | shuffle | sigma | Valid. Acc.
---------- | ----------- | ----------- | ------ | ------------ | ------------ | ----------- | --------- | ----------- | -- | ------------- | ---- | ------- | ----- | -----------
128 | 6 | 16 | 10 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.882312924
128 | 6 | 16 | 10 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.005 | TRUE | 0.1 | 0.929931973
128 | 6 | 16 | 10 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.01 | TRUE | 0.1 | 0.898639456
256 | 6 | 16 | 10 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.005 | TRUE | 0.1 | 0.920181406
256 | 6 | 16 | 10 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.886621314
512 | 6 | 16 | 10 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.866439912
512 | 6 | 16 | 20 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.877097503
256 | 6 | 16 | 20 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.890702948
128 | 6 | 16 | 20 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.897052154
64 | 6 | 16 | 20 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.901587302
64 | 6 | 16 | 20 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.005 | TRUE | 0.1 | 0.933106576
64 | 16 | 32 | 20 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.005 | TRUE | 0.1 | 0.929251701
128 | 16 | 32 | 20 | 120 | 84 | 5 | TRUE | 1 | 0 | FALSE | 0.005 | TRUE | 0.1 | 0.93356009
128 | 16 | 32 | 20 | 240 | 120 | 5 | TRUE | 1 | 0 | FALSE | 0.005 | TRUE | 0.1 | 0.939682539
128 | 16 | 32 | 20 | 240 | 120 | 5 | FALSE | 3 | 0 | FALSE | 0.005 | TRUE | 0.1 | 0.852154195
128 | 16 | 32 | 20 | 240 | 120 | 5 | TRUE | 1 | 0 | TRUE | 0.005 | TRUE | 0.1 | 0.054421769
128 | 16 | 32 | 20 | 240 | 120 | 5 | TRUE | 1 | 0 | FALSE | 0.005 | TRUE | 0.1 | 0.932199546
128 | 16 | 32 | 20 | 240 | 120 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.931065759
128 | 32 | 64 | 20 | 240 | 120 | 5 | TRUE | 1 | 0 | FALSE | 0.001 | TRUE | 0.1 | 0.959410431


The first parameter that I tuned was the learning rate, just to make sure the network was not training too fast or too slow, which is easy to notice.
Then I played with the depth of the convolution layers and the size of the fully connected layers.
My reasoning here was to make the depth of the convolution layers and the fully connected layers bigger than LeNet to take into account the larger number of classes. My belief was that the network would need more feature maps to differentiate the different signs.
This seems to be true as the network got better with deeper convolutions.
, and also the different pre-processing techniques.


My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.2% 
* test set accuracy of 94.2%

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="/Visualizations/web_image_set/modified/traffic_sign_1_edited.jpg" width="120" height="120">
<img src="/Visualizations/web_image_set/modified/traffic_sign_2_edited.jpg" width="120" height="120">
<img src="/Visualizations/web_image_set/modified/traffic_sign_3_edited.jpg" width="120" height="120">
<img src="/Visualizations/web_image_set/modified/traffic_sign_4_edited.jpg" width="120" height="120">
<img src="/Visualizations/web_image_set/modified/traffic_sign_5_edited.jpg" width="120" height="120">

Those images should be predicted correctly by the network. They seem to present no particular problem.
I cropped them to a square size to format them similarly to the training data.

Then my code resize them to 32x32x3 before applying the same pre processing steps as the training data, and feeding them to the network for classification.

####2. Discuss the model's predictions

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		| Road Work  									| 
| General Caution     	| General Caution 								|
| Right-of-way		    | Right-of-way									|
| No Entry  			| No Entry      					    		|
| 30 km/h	      		| 30 km/h					 				    |

The model did very well on this test set. 100% prediction. Since we do get a 94% accuracy, this is expected on such a small dataset.

####3. Describe how certain the model is when predicting on each of the five new images.

For each of the 5 images, here is how certain the code is in predicting the different traffic signs:
```
-----------------------------------------
Top Probabilities for image 1:
  Road work: 100.00 %
  General caution: 0.00 %
  Dangerous curve to the right: 0.00 %
  Bicycles crossing: 0.00 %
  Turn right ahead: 0.00 %
-----------------------------------------
Top Probabilities for image 2:
  General caution: 100.00 %
  Right-of-way at the next intersection: 0.00 %
  Pedestrians: 0.00 %
  Go straight or left: 0.00 %
  Traffic signals: 0.00 %
-----------------------------------------
Top Probabilities for image 3:
  Right-of-way at the next intersection: 100.00 %
  Beware of ice/snow: 0.00 %
  End of no passing: 0.00 %
  Pedestrians: 0.00 %
  Dangerous curve to the right: 0.00 %
-----------------------------------------
Top Probabilities for image 4:
  No entry: 100.00 %
  Traffic signals: 0.00 %
  Turn right ahead: 0.00 %
  Beware of ice/snow: 0.00 %
  Turn left ahead: 0.00 %
-----------------------------------------
Top Probabilities for image 5:
  Speed limit (30km/h): 100.00 %
  Speed limit (80km/h): 0.00 %
  Speed limit (50km/h): 0.00 %
  Speed limit (20km/h): 0.00 %
  Speed limit (70km/h): 0.00 %
```

Nothing much to say here, for each of the signs, there is just one clear winner.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


