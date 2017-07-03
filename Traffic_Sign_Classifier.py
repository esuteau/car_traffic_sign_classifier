
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Step -1: Download the dataset

# In[1]:

from urllib.request import urlretrieve
from os.path import isfile, isdir
from zipfile import ZipFile
from tqdm import tqdm
import sys

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


# ---
# ## Step 0: Load The Data

# In[2]:

# Load pickled data
import pickle
import os

# Verify the files were correctly extracted
dirs = os.listdir(base_path)
print(dirs)

training_file = 'train.p'
validation_file = 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[3]:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation =  X_valid.shape[0]

# Number of testing examples.
n_test =  X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
print("Training Set: {}, Size: {}".format(n_train, X_train.shape))
print("Validation Set: {}, Size: {}".format(n_validation,  X_valid.shape))
print("Test Set: {}, Size: {}".format(n_test,  X_test.shape))
print("Image data shape: {}".format(image_shape))
print("Number of classes: {}".format(n_classes))


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[4]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random

# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')


# ### Parse the sign names from the csv

# In[5]:

sign_names = {}
with open('signnames.csv', 'r') as f:
    file_data = f.readlines()
    
for n, line in enumerate(file_data):
    if n == 0:
        continue
    res = line.split(',')
    sign_names[int(res[0])] = res[1].strip('\n')


# ### Example of input image

# In[6]:

# Show a traffic sign from the training set.
index = 500
image_data = X_train[index]

print('Image Label: {}: {}'.format(y_train[index], sign_names[y_train[index]]))
print('Image Shape: {}'.format(image_data.shape))

if image_data.shape[2] == 1: 
    plt.imshow(image_data.squeeze(), cmap='gray')
else:
    plt.imshow(image_data.squeeze())


# ### Distribution of the Training Set

# In[7]:

import numpy as np

# Look at the distribution of the different labels in the training set.
# Normalize it, so that it looks like probabilities.
n, bins, patches = plt.hist(y_train, n_classes, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Labels')
plt.ylabel('Probability')
plt.title('Label Distribution for the Training Set')
plt.axis([0, n_classes, 0, 0.06])
plt.grid(True)
plt.show()

# Look at the different between the most common traffic sign and the least common
most_common = np.max(n)
least_common = np.min(n)
most_least_ratio = most_common / least_common
print('Most common sign has {} more data samples than least common'.format(most_least_ratio))


# ### Distribution of the Validation set

# In[8]:

n, bins, patches = plt.hist(y_valid, n_classes, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Labels')
plt.ylabel('Probability')
plt.title('Label Distribution for the Validation Set')
plt.axis([0, n_classes, 0, 0.06])
plt.grid(True)
plt.show()


# ### Distribution of the Test set

# In[9]:

n, bins, patches = plt.hist(y_test, n_classes, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Labels')
plt.ylabel('Probability')
plt.title('Label Distribution for the Test Set')
plt.axis([0, n_classes, 0, 0.06])
plt.grid(True)
plt.show()


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[10]:

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import numpy as np
from sklearn.utils import shuffle
from skimage.color import rgb2gray
from skimage.transform import resize

def preprocess(images, hyperparams = {}):

    image_new = images

    if hyperparams['greyscale']:
        #print('Average: {}'.format(np.average(image_new.reshape((1,-1)))))
        print('Converting Dataset to Grayscale and Normalizing')
        image_new = rgb2gray(image_new).reshape(image_new.shape[0], image_new.shape[1], image_new.shape[2], 1)
        #print('Average: {}'.format(np.average(image_new.reshape((1,-1)))))

    if hyperparams['normalization']:
        print('Normalizing the data to zero mean')

        # Show Data Before Normalization, and After
        print('Data before pre-processing: {}'.format(image_new[index].reshape((1,-1))))

        # Normalize Data (Doesn't help)
        image_new = (image_new - 128) / 128

        print('Data after pre-processing: {}'.format(image_new[index].reshape((1,-1))))
        print('Average: {}'.format(np.average(image_new.reshape((1,-1)))))
        
    return image_new


# Pre process the input data
hyperparams = {}

hyperparams['normalization'] = False
hyperparams['shuffle'] = True
hyperparams['greyscale'] = True

# Reload the dataset
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

X_train = preprocess(X_train, hyperparams)
X_valid = preprocess(X_valid, hyperparams)
X_test = preprocess(X_test, hyperparams)


# In[11]:

# Plot the same example after pre-processing
image_data = X_train[index]

print('Image Label: {}'.format(y_train[index]))
print('Image Shape: {}'.format(image_data.shape))

if image_data.shape[2] == 1: 
    plt.imshow(image_data.squeeze(), cmap='gray')
else:
    plt.imshow(image_data.squeeze())


# In[12]:

# Shuffle Data
if hyperparams['shuffle'] :
    print('Shuffling Data')
    X_train, y_train = shuffle(X_train, y_train)


# In[13]:

# After Pre-processing, display the dataset size again
print("Training Set: {}".format(X_train.shape))
print("Validation Set: {}".format(X_valid.shape))
print("Test Set: {}".format(X_test.shape))
print("Number of classes: {}".format(y_train.shape))


# ### Model Architecture

# In[14]:

### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten

def LeNetTrafficSigns(x, parameters = {}):  
    
    # We base this classifier on LeNet's architecture.
    
    # Get the network parameters from a dictionary
    mu = parameters.get('mu', 0)
    sigma = parameters.get('sigma', 0.1)
    input_depth = parameters.get('input_depth', 3)
    filter_size = parameters.get('filter_size', 5)
    conv1_depth = parameters.get('conv1_depth', 6)
    conv2_depth = parameters.get('conv2_depth', 16)
    fc1_out_size = parameters.get('fc1_out_size', 120)
    fc2_out_size = parameters.get('fc2_out_size', 84)
    
    print('Network Parameters')
    print('mu: {}'.format(mu))
    print('sigma: {}'.format(sigma))
    print('input_depth: {}'.format(input_depth))
    print('filter_size: {}'.format(filter_size))
    print('conv1_depth: {}'.format(conv1_depth))
    print('conv2_depth: {}'.format(conv2_depth))
    print('fc1_out_size: {}'.format(fc1_out_size))
    print('fc2_out_size: {}'.format(fc2_out_size))
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x16.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, input_depth, conv1_depth), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(conv1_depth))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, conv1_depth, conv2_depth), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(conv2_depth))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x32. Output = 800.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 800. Output = 120.
    fc1_in_size = filter_size * filter_size * conv2_depth
    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_in_size, fc1_out_size), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(fc1_out_size))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(fc1_out_size, fc2_out_size), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(fc2_out_size))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(fc2_out_size, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


# In[15]:

## Hyper parameters
epochs = 20
batch_size = 128
rate = 0.001

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


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[16]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

import tensorflow as tf

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Define Tensorflow variables
x = tf.placeholder(tf.float32, (None, 32, 32, hyperparams['input_depth']))
y = tf.placeholder(tf.int32, (None))

# One-Hot encode outputs
one_hot_y = tf.one_hot(y, n_classes)

# Create the Model
logits = LeNetTrafficSigns(x, hyperparams)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Define Accuracy Metrics
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# Train the Network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        training_accuracy = evaluate(X_train, y_train)    
        validation_accuracy = evaluate(X_valid, y_valid)
        print("Epoch {}/{}, Training Accuracy = {:.3f},  Validation Accuracy = {:.3f}".format(i+1, epochs, training_accuracy, validation_accuracy))
        
    saver.save(sess, './trafficSignClassifier')
    print("Model saved")
    
# Save the Results and parameters to a file, so that we can compare different architectures.
with open("results.txt", "a") as myfile:
    keys_sorted = sorted(hyperparams.keys())
    for key in keys_sorted:
        myfile.write('{}\t'.format(key))
    myfile.write('Valid. Acc.\n')
    
    for key in keys_sorted:
        myfile.write('{}\t'.format(hyperparams[key]))
    myfile.write("{}\n".format(validation_accuracy))


# In[17]:

# Evaluate the Model on the test set
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[27]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
from skimage.color import rgb2gray
from skimage.data import load
from skimage.transform import resize

# Load the images and resisze them as 32x32x3 inputs.
test_images_folder = base_path + '/Visualizations/web_image_set/modified/'
list_images = sorted(os.listdir(test_images_folder))
print(list_images)

new_images = np.zeros((len(list_images), 32, 32, 3))
new_labels = [25, 18, 11, 17, 1] #Defined manually

for n, image in enumerate(list_images):
    # Load the image
    image_data = load(test_images_folder + image)
    
    # Resize Image to 32x32x3
    image_resized = resize(image_data, (32, 32), mode='reflect')
        
    # Store image data
    new_images[n,:,:,:] = image_resized
    
# Display all images
for n in range(len(new_images)):
    plt.subplot(1, len(new_images), n+1)
    plt.imshow(new_images[n])
    
plt.show()

print('New Image dataset size: {}'.format(new_images.shape))


# ### Predict the Sign Type for Each Image

# In[24]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

import tensorflow as tf

saver = tf.train.Saver()

# Pre process the data
new_images_proc = preprocess(new_images, hyperparams)
softmax_prob = tf.nn.softmax(logits)
top_outputs = tf.nn.top_k(softmax_prob, 5)

# Calculate the logits for each image.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    # Calculate the top 5 K predictions
    outputs = sess.run(top_outputs, feed_dict={x: new_images_proc})

# Extract Values and Indices
top_values = outputs.values
top_indices = outputs.indices

# Predicted Labels
pred_labels = [ind[0] for ind in top_indices]
for n, pred in enumerate(pred_labels):
    plt.figure()
    plt.title('Label: {}, {}'.format(pred, sign_names[pred]))
    plt.imshow(new_images[n])
    
plt.show()


# ### Analyze Performance

# In[31]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
pred_performance = 100 *np.sum(np.equal(new_labels, pred_labels)) / len(new_labels)
print('Test Performance: {:.2f} %'.format(pred_performance))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[34]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

for n in range(0, len(new_labels)):
    print('-----------------------------------------')
    print('Top Probabilities for image {}:'.format(n+1))
    top_prob = top_values[n]
    top_ind = top_indices[n]
    for n_p in range(0, len(top_prob)):
        print('  {}: {:.2f} %'.format(sign_names[top_ind[n_p]], top_prob[n_p] * 100))


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[22]:

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

