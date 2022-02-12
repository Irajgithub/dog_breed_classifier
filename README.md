## dog breed classifier
This work is a capstone project for the Udacity Data Science Nano Degree. A Convolutional Neural Network (CNN) is trained to create an algorithm for recognizing a dog's breed from an input image. This classifier method determines if a dog or a human is present in the image, then distinguishes the dog's breed if a dog is there and returns the resembling dog breed for that person if a human is detected. Isn't that amusing!?
The project is divided into eight steps.

[Step 1. Import Datasets](#Step-1)

[Step 2. Detect Humans](#Step-2)

[Step 3. Detect Dogs](#Step-3)

[Step 4. Create a CNN to Classify Dog Breeds (from Scratch)](#Step-4)

[Step 5. Use a CNN to Classify Dog Breeds (using Transfer Learning)](#Step-5)

[Step 6. Create a CNN to Classify Dog Breeds (using Transfer Learning)](#Step-6)

[Step 7. Write the Algorithm](#Step-7)

[Step 8. Test the Algorithm](#Step-8)


### Step 1. Import Datasets<a name="Step-1"></a>
The data is imported as a dataset of dog photos in the first stage. The data set is divided into three categories using the scikit-learn library's load-files function: train-files, valid-files, test-files, which are numpy arrays containing image file paths, and train-targets, valid-targets and test-targets, which are numpy arrays containing on-hot-encoded classification labels for 133 different dog breeds. And in the following, a dataset of human images is imported where the file paths are stored in the numpy array named human-files.

### Step 2. Detect Humans<a name="Step-2"></a>
Instead of developing a model, OpenCV's implementation of "Haar feature-based cascade classifiers" is used to recognize humans in images. Many pre-trained face detectors of OpenCV library are available on github as XML files. One of these detectors has been downloaded and stored it in the haarcascades directory. Before using the face detectors, it is standard procedure to convert the images to grayscale. The detectMultiScale function uses the grayscale picture as a parameter to run the classifier contained in face-cascade. Each identified face is represented as a one-dimensional array with four entries, each of which defines the detected face's bounding box. The first two elements in the array (extracted as x and y in the code) describe the horizontal and vertical coordinates of the bounding box's top left corner. The width and height of the box are specified by the final two values in the array (extracted in the code as w and h). This technique is used to create a function that returns True if a human face can be recognized in a picture and False if it cannot.

### Step 3. Detect Dogs<a name="Step-3"></a>
A pre-trained ResNet-50 model is used to recognize dogs in photos in this part. This picture categorization is based on the "ImageNet" dataset, which is a very big and well-known dataset. Over ten million URLs in ImageNet point to a picture containing an item from one of 1000 categories. In this section, the path-to-tensor function takes a string-valued file path to a color picture as input and generates a 4D tensor appropriate for feeding into a Keras CNN. The code initially loads the picture and resizes it to 224×224 pixels in square format. The image is then transformed to an array before being scaled to a 4D tensor. Because we're working with color photos, each image has three channels in this situation. Similarly, because we're dealing with a single picture (or sample), the returned tensor will always be of the shape (1,224,224,3).

The paths-to-tensor function accepts a numpy array of string-valued picture paths and outputs a 4D tensor with the shape (number of the samples,224,224,3). Additional processing is required to prepare the 4D tensor for ResNet-50 and any other pre-trained model in Keras. By rearranging the channels, the RGB picture is first transformed to BGR. All pre-trained models have the additional normalization step that the mean pixel must be subtracted from every pixel in each image. This is implemented in the imported function preprocess-input.

Now that we have a way to format our image for supplying to ResNet-50, we are ready to use the model to extract the predictions. This is accomplished with the predict method, which returns an array whose 𝑖i-th entry is the model's predicted probability that the image belongs to the 𝑖i-th ImageNet category. This is implemented in the ResNet50-predict-labels function in the code.

By taking the argmax of the predicted probability vector, one can obtain an integer corresponding to the model's predicted object class, which we can identify with an object category using a dictionary. The categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all dog categories. Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the ResNet50-predict-labels function returns a value between 151 and 268 (inclusive). These ideas are used to complete the dog-detector function, which returns True if a dog is detected in an image and False if not.




### Step 4. Create a CNN to Classify Dog Breeds (from Scratch)<a name="Step-4"></a>
Now that we have functions for detecting humans and dogs in an image, we need a way to predict breed from images. It should be noted that the task of assigning breed to dogs from images is considered exceptionally challenging. To see why, consider that even a human would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel. In this step, a CNN is created that classifies dog breeds. It was attempted to achieve a test accuracy of at least 1%. Because there are 133 classes, the accuracy greater than 1% indicates that the model is doing better than a random guess. The trained CNN model gained above 12% accuracy on the test data set.

### Step 5. Use a CNN to Classify Dog Breeds (using Transfer Learning)<a name="Step-5"></a>
To reduce training time without sacrificing accuracy, transfer learning is used to train a CNN. The model employs the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to next layer of the model. A global average pooling layer is added and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax activation function.


### Step 6. Create a CNN to Classify Dog Breeds (using Transfer Learning)<a name="Step-6"></a>
This stage involves using transfer learning to develop a CNN that can identify dog breeds from photos, with the goal of achieving at least 60% accuracy on the test set. This step uses a pre-trained Xception CNN model. The prediction function, similar to the function in Step 5, comprises three steps:
1.	Extract the bottleneck features corresponding to the chosen CNN model.
2.	Supply the bottleneck features as input to the model to return the predicted vector.
3.	Use the dog-names array defined in Step 0 of this notebook to return the corresponding breed.
The model in this step shows accuracy around 84% on the test set


### Step 7. Write the Algorithm<a name="Step-7"></a>
An algorithm is assembled here to accept a file path to an image and first determines whether the image contains a human, dog, or neither. Then,

•	if a dog is detected in the image, return the predicted breed.

•	if a human is detected in the image, return the resembling dog breed.

•	if neither is detected in the image, provide an output that indicates an error.

### Step 8. Test the Algorithm<a name="Step-8"></a>
In this section, the algorithm is tested on some random images to see how it works. It appears that the model is unable to distinguish humans well. For that, we'll need to create a model and feed the model with a lot of data, including images of people from various perspectives. The model does a decent job at distinguishing dogs and their breeds. It is not confused with cats, which is a good thing. Another issue with the model is that it does not operate with drawings, which may be addressed by training the model using data that includes drawing pictures.

