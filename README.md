# Automatic Image Recognition Classifier

** This is a work in progress **
** google_images_download works best when the image count is 500 or less. It is possible to make multiple requests from different time periods and combine the results in order to get larger sample sizes for each class. **


## Problem Statement:

Creating a python scrypt that gathers images and processes them accordingly. Then creates and analyzes a multi-class classification model with those images, automatically.

## Data Collection

The google_images_download package was the main tool used for data collection. It allows the program to search for whatever images the user sets up as the topics. With the way it is set up right now, it allows for a maximum of 500 images, that were uploaded in 2018, for each class. But it is possible to add more images, by adding new requests for different years.

The images are automatically organized into a 'data' folder and then subfolders for each class.

When the 'model_tester' function is called, this downloads 3 images from each class, to use as validation data and test predictions.

## Data Processing

The images are processes using cv2, transforming all of them into a numpy array of 4 dimensions, including the total number of images, the pixel size, and an RGB structure.

An Image Data Generator was also used when fitting the model, which generates new images with small tweaks, slightly altering them in different ways, to prevent the model from using the position of items in the pictures as a feature.

## Program Structure

There are a total of 3 functions: image_collector, model_maker, and model_tester. The program is split up in these three functions to allow more flexiility and not forcing the user to gather the same data again in case they want to change the number of epochs they want to use, for example.

The output of the program includes a chart with the increase/decrease of loss and accuracy over each epoch, and predictions made on 3 images, one for each class, with the probability of each image belonging to each class.


## Resources

https://keras.io/examples/cifar10_resnet/

https://opendatascience.com/building-a-convolutional-neural-network-male-vs-female/

https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24

https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

