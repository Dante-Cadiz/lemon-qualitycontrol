# Lemon Quality Control
Lemon Quality Control is a complete data science and machine learning project with the business goal of differentiating good and bad quality lemons. This project is delivered to the client via a [Streamlit dashboard](https://lemon-quality-control.herokuapp.com/) in which the client can upload images to apply the project's binary classification machine learning model to in order to predict the lemon's quality, along with pages containing the findings from the project's conventional data analysis, a description and analysis of the project's hypotheses, and performance evaluation of the machine learning model. 
The project also contains a functional pipeline in the form of three Jupyter notebooks, which cover initial data importation and cleaning, data visualisation, and development and evaluation of the project's TensorFlow deep learning model respectively.

## Dataset Content
* This project uses 2 datasets from separate sources that are later combined and analysed simultaneously to fulfil the project's business objectives.
* The first dataset for this project is a large collection of 2D images of lemons in a zipped file, pre-labelled as either being of good or poor quality based on their condition. There is also a third category of images; blank backgrounds not containing a lemon. I have chosen to ignore this third category here as analysing it is not necessary to fulfil the project's business requirements. These images are all 300x300px but will be resized during the image preparation process to allow for more time-efficient model learning. This dataset was found here on [Kaggle](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset).
* The second dataset is another large collection of 2D images, annotated manually by the data owner using OpenCV's CVAT. The annotations capture typical lemon quality issues such as illness, gangrene, mould, visual blemishes, and dark style remains. The annotations also cover structural regions of the lemon and photographic features such as lighting issues or blur. The annotated dataset can be accessed, read and viewed via [pycocotools' COCO function](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py). These images are all 1056x1056px but will be resized during the image preparation process to allow for more time-efficient model learning. This dataset was found here on [Kaggle](https://www.kaggle.com/datasets/maciejadamiak/lemons-quality-control-dataset).
* This second dataset requires a large amount of cleaning, as well as making the arbitrary decision of which defects should classify as a 'bad quality' lemon for binary classification purposes. This decision was difficult and required both careful visual analysis of the dataset itself and industry research such as looking at the UK Government's [Specific Marketing Standards](https://www.gov.uk/guidance/comply-with-marketing-standards-for-fresh-fruit-and-vegetables) for citrus fruits. The UK Government uses a 'class' system to grade lemons' quality. For the purposes of this project, I decided that this factory would seek to only provide lemons of Class 1 and above. While this project does not provide an exact evaluation of the lemon's 'class' in this regard, it does serve as an effective approximation and identifier of traits that would immediately disqualify a lemon as being Class 1 or higher. The eventual chosen determining factors for description of a lemon as bad quality were as follows: gangrene, mould, blemishes, and images tagged as having general non-specific illness.
* When combining two different image datasets that have different camera angles, backgrounds, and ratio of foreground to background elements, considerable data cleaning is required to standardise them against each other. To do this, I took the following steps: applying the [Rembg](https://github.com/danielgatis/rembg) package to isolate the foreground element of the lemon itself and remove the image's background, then getting the bounding box of the lemon using [PIL's Image module](https://pillow.readthedocs.io/en/stable/reference/Image.html) and cropping the image to the dimensions of the bounding box. Thus the standard image of a lemon taking majority space against a minimal blank background was developed.

## Business Requirements
* The stakeholder for this project is the owner of a small-scale commerical fruit preparation factory that supplies local markets. 
* The factory handles and processes fruit delivered from local farms to ensure it is ready for supermarket display and sale. 

- The factory's current process for separating its lemons into those suitable and unsuitable for retail currently involves: 
- Conveyor belt machinery that measures diameter of lemons to check it is within acceptable bounds
- Performing chemical tests such as ripening gas concentration, determination of percentage juice content, and assessment for common types of rot
- Factory employees manually inspecting and separating the fruits based on a visual assessment of blemishes and overall skin quality. This inspection is time-consuming and volatile in terms of employee mistakes, and requires extensive employee training, and is as such not scalable to a larger sphere. 
- The factory owner wants to eliminate this final process and replace it with an ML process that instantly assesses lemon images, speeding the overall preparation of lemons up and removing reliance on individual human judgement and susceptibility to error. If this process is assessed as successful and efficient for lemons, it could be extended to the factory's preparation of other fruits, thus improving overall output efficiency greatly.

- As such, the project has the following business objectives:
- The client is interested in analysing the visual difference between good and poor quality lemons, specifically the visual markers that define a poor quality lemon. This will be satisfied via conventional data analysis: finding the average image and variability per label in the data set, as well as the contrast between said labels.
- The client is interested in accurately and instantly predicting from a given image whether a lemon is of good or poor quality. This business objective will be delivered by the development and deployment of a TensorFlow deep learning pipeline which serves to perform a binary classification task on the lemon images.
- The content satisfying these two business objectives will be combined into a Streamlit dashboard and delivered to stakeholders.

## Rationale to map the business requirements to the Data Visualizations and ML tasks

Business Requirement 1:
- As a client, I can navigate easily around an interactive dashboard so that I can view and understand the data presented.
- As a client, I can view and toggle visual graphs of average images (and avergae image difference) and image variabilities for both good and bad quality lemons so that I can observe the difference and understand the visual markers that indicate lemon quality better.
- As a client, I can view an image montage of either good or bad quality lemons so that I can visually differentiate them.

Business Requirement 2:
- As a client, I can access and use a machine learning model so that I can obtain a class prediction on a lemon image provided.
- As a client, I can provide new raw data of a lemon and clean it so that I can run the provided model on it.
- As a client, I can feed cleaned data to the dashboard to allow the model to predict on it so that I can instantly discover whether a given lemon is of good or poor quality.
- As a client, I can save model predictions in a timestamped CSV file so that I can keep an account of the predictions that have been made.

## Hypothesis and how to validate?
* The project's initial hypothesis was for each business objective as follows:
* We suspect that there would be a clear and significant visual difference noticeable between the average and variability images for each label, both in colour and texture, with bad quality lemons presenting areas of discoloration and contouring compared to smooth and plain yellow good quality lemon average images.
* This hypothesis will be validated through visual examination of the generated images, along with application of Sobel filtering to bring out defined areas of edges in the images and clearly illuminate these differences.
* Providing a new lemon image and applying the project's generated binary classification model to it would allow the client to predict the likely quality of a lemon to a high degree of accuracy.
* This hypothesis will be validated through the testing and graphical evaluation of the generated model, specifically logging its validation accuracy and loss between epochs, as well as creating a confusion matrix between the two outcomes.
* If these two hypotheses are validated, the client can use the insights offered by conventional image data analysis and the model to perform quality control tasks on their lemons with a high degree of accuracy.

## ML Business Case 
- The client is interested in accurately predicting from a given image whether a lemon is of good or poor quality. This business objective will be delivered by the development and deployment of a TensorFlow deep learning pipeline trained on the previously generated dataset of class-labelled lemons.
- This TensorFlow pipeline is a convolutional neural network, the form of neural network that is generally best suited to identifying patterns and areas of importance in image arrays, specifically through its convolution and pooling layer pairs.
- The eventual target of the machine learning pipeline is a binary classification single label model. Its ideal outcome is a model that can successfully predict the class of a lemon image as good or poor quality.
- The model's output is defined as a flag that signals the quality of a lemon based on the associated probability of its quality as generated by the model. 
- After the outcome is generated, the heuristic is as follows: Lemons deemed as bad quality are then discarded without further human examination and those determined to be good quality are sent for further review under the processes previously described in the business requirements section.
- The stakeholder has decided that the most important metrics for evaluating the success of the machine learning model are overall model accuracy (evaluated via F1 score) and recall on the negative class result of a poor quality lemon. 
- The minimum threshold for accuracy as defined by the stakeholder is an F1 score of 0.95. A model that has a high degree of accuracy and as such a low amount of mistakes made in classifying lemons will allow the factory to make maximal profit by improving overall product average quality as well as reducing required man hours in assessing fruits.
- Recall on poor quality lemons is viewed as specifically important to the client because providing possibly poor quality lemons to supermarkets and other subsequent companies involved in the supply chain is a worse outcome than mistakenly discarding good quality ones. A preliminary threshold for recall on bad quality lemons to assess the model's success has been set at 0.98.
- As such, a successful model that fulfils the business requirements is one that has an F1 score of 0.95 or higher and a recall on bad quality lemons of 0.98 or higher.


## Development and Machine Learning Model Iterations

* The Tensorflow binary image classification model went through a series of iterations and hyperparameters in order to produce an optimised model capable of handling the data.
* Version 5 was the eventual accepted production version. Legacy versions of the models and their performance can be found in the 'Outputs' section of this repository.
* The general structure of the model was heavily influenced by Code Institute's Malaria Detector model for binary classification, and adjusted and tested upon to find optimal hyperparameters.
* Before the model trained on it, data was augmented using TensorFlow's ImageDataGenerator, which stretched, flipped, and rotated the images in order to provide a larger dataset and also avoid the issue of natural light in photo camera angles affecting the model's performance.
* The model was constructed of three consecutive pairs of 2D convolution and pooling layers used to isolate areas of importance and contrast within the image for the model to train itself on. The numbers of filters for the three convolution layers were set to values of 32, 64, and 64 respectively and the kernel sizes at (3,3). Powers of 2 were chosen as filter numbers for the convolution layers to optimise processing. Pooling layer pool sizes were set to the industry standard value of (2,2).
* This is followed by a single dense layer of many neurons (128 in the final production model), a dropout layer of 0.5 in order to avoid overfitting of the model, and a final dense layer containing a single neuron with a sigmoid activation function, as is standard for binary classification models. The sigmoid activation function was chosen after consulting [this information](https://www.geeksforgeeks.org/activation-functions-neural-networks/)
* Before running each iteration of the model, the most suitable hyperparameters out of a user-provided selection were attained using the Keras Tuner.
* Two hyperparameters were selected in this search process for optimisation; the neuron count in the main densely connected layer of the neural network (varied between 64 and 512 with a step of 64 after V1), and the learning rate of the model (set to either 0.001 or 0.0001). These values were chosen with guidance from [TensorFlow's Keras Tuner Tutorial](https://www.tensorflow.org/tutorials/keras/keras_tuner) and later simplified to remove those hyperparameter combinations that performed consistently poorly in order to increase tuning speed.
* The hyperparameter optimisation search was conducted using the Hyperband search algorithm. This algorithm was chosen after reading [this article](https://2020blogfor.github.io/posts/2020/04/hyperband/) with the goal of efficiency and fast development compared to more time-intensive Bayesian optimisation algorithms.
* The data was loaded into the model in batch sizes of 20, a relatively small even number which was chosen both to further improve time efficiency of the model and avoid overfitting.
* TensorFlow's EarlyStopping function was included to halt training of the model early when the loss value on validation data was no longer clearly improving (a 'patience' value of 3 was passed in).
* The model was then trained over a possible 25 epochs with steps per epoch set to the train set class lengths divided by 20 (the batch size). Previously defined class weights and early stopping function were passed into the fit() function.

* V1 -This model was only applied to the initial smaller uncombined single lemon dataset, with the images at their original average size (300, 300). While this model predicted classes with a great degree of accuracy (over 98%), the dataset was limited in size and expansion was required.

* V2 - This iteration of the model incorporated the second lemon image dataset. Gangrene, mould, blemishes, and general illness were chosen as the markers which determined a poor quality lemon.
The combined dataset was not preprocessed or standardised in any way, the only change being resizing the images to (50, 50) to speed up model training.
The introduction of the combined dataset led to a greater target imbalance between classes.
The model hyperparameter search was adjusted to a different range of possible neuron counts in the model's main dense layer, namely 64 to 512 with a step factor of 64. 
This model performed very poorly with the new data addition only reaching 82% validation accuracy. 
The model's recall was excellent on bad quality lemons but very poor on good quality lemons, in fact below 60%.
The hypotheses for why the model performed poorly were as follows:
Due to the introduction of the second dataset which contained majority bad quality lemons, the bad quality images were heavily made up of smaller lemons against large empty backgrounds, as opposed to the first dataset which contained lemons that took up the majority of the images and were set against textured wallpaper backgrounds.
As such I hypothesised that the model may have identified this background difference and relative lemon size contrast as a determining factor for selecting the probable class of a lemon image. I also hypothesised that the reduction in image dimensions potentially had erased important information in determining the classes.
I also considered that the unchecked target imbalance may have made a difference, as the model heavily favoured predicting the majority class. 

* V3 - For this iteration of the model, I adjusted the criteria in the second image dataset that marked a poor quality lemon to only gangrene and mould (the two features that would immediately rule a lemon out of contention for any commerical use). 
To avoid the pitfalls of the model's previous iteration, I took extensive data cleaning steps to preprocess and standardise the images to ensure that the model predicted on relevant features. 
I applied Rembg's background removal package to all images to obtain only the foreground element of the lemon itself and erase the background content (either wallpaper or an empty black background depending on the image's original dataset).
This process was followed by applying the PIL package to get the lemons' bounding boxes in their image and subsequently cropping the images to the outer sizes of the bounding boxes to fully clean the image data. 
I then reduced the image size to (100, 100) for a compromise between fast model training/hypertuning and accurate learning of patterns. 
Despite my hypothesis that this initial data cleaning step would lead to a marked improvement, the model itself was not accurate in predicting the class of an image and performed no better than the second iteration, in fact worse. 
Once again, the recall on bad quality lemons was successful but very poor on good quality lemons.
As such, I hypothesised that the target imbalance of outcomes was significant.

* V4 - The process for the fourth and final iteration of the model began with changing the criteria for a bad quality lemon in the second dataset to all of gangrene, mould, blemishes, and general illness. 
This increased target imbalance greatly once again, so I considered a variety of options to remedy this, namely resampling of the minority class, undersampling of the majority class, performing SMOTE on the minority class, and adding skewed class weights to the model itself. I chose not to resample the minority class to avoid overfitting, and undersampling of the majority class was also undesirable due to the already somewhat limited nature of the dataset. Finally, SMOTE was considered unviable because of the lengthy time and computation process required to perform SMOTE on the NumPy arrays of the images.
As such, I incorporated Scikit's utility for calculating class weights for the model to favour the minority class, and passed these calculations into the model using the class_weights hyperparameter, then performed hyperparameter optimisation with the Keras Tuner using these class weights, and ran the model with the same class weights. 
This iteration of the model performed successfully, with an average F1 score of 0.96, as well as 99% recall on the bad quality class. It exceeded the predetermined thresholds for both recall on bad quality lemons and overall accuracy. 

* V5 - Late in development, I realised a possible issue with the previous models, relating to the dataset itself. Upon analysing average images, I noticed circular patterns in the centre of the average image for those lemons labelled bad quality. I had initially hypothesised that these circles signified blemishes, but later I came to hypothesise that these may have in fact been the pedicels of the lemons. The pedicel of the lemon is simply the part of the fruit which once connected it to the tree it grew on, and not an indicator of quality one way or another, and as such having the model possibly identifying pedicel areas of images as a factor to predict class on was problematic. The second dataset which contained majority lemons labelled as bad quality was different to the first dataset in that it contained lemons photographed from many angles, including those in which the pedicel was pointed frontward in the direction of the camera. The first dataset only contained images in which the pedicel was at the side or obscured entirely. As such, I added a further data cleaning step to remove the images from the second dataset that were tagged as having a prominent pedicel, and combine those that were not with the first dataset. 
This additional data cleaning step proved effective both in reducing the potential influence of pedicel foreground presence on model prediction (average image plots of the bad quality class no longer displayed the circular markings that I had anticipated were due to the presence of pedicels), and the overall accuracy and recall on poor quality lemons of the resultant model. The hyperparameter optimisation search suggested that the optimal model configuration was a dense layer of 128 neurons and a learning rate of 0.001. I then trained the suggested model and evaluated it on the test set. The result was an overall accuracy of 0.9646, along with a rounded recall score of 0.98 on the bad quality class. As it satisfied both metric success criteria as defined by the stakeholder, this model was accepted as the production model.


## Dashboard Design and Features

### General application design and home page

* This project is delivered via a Streamlit dashboard web application containing five app pages. The client can easily navigate between these pages via the interactive menu present on the left of the page, as seen below.
* The project's home page contains a basic summary of the business process driving this project's creation, along with links to further documentation about the project.
![Full page layout with menu and project summary](https://i.imgur.com/ENa74AP.png)

### Presentation of data visualisation plots

* The Lemons Visualiser page covers the first Data Analysis business objective of the project. It contains toggle-able plots that can easily be opened and closed via the inbuilt toolbar, examples of which are below.
![Average and variability of images example](https://i.imgur.com/EyEADjt.png)
![Difference between average and variability of images](https://i.imgur.com/3D4JndY.png)
* This app page also contains an image montage creation tool, where the user can slect a class of label for which to display a montage generated via graphical presentation of random validation set images.
![Image montage creation tool](https://i.imgur.com/iDcbtOS.png)

### Lemon Quality Assessor tool

* The Lemon Quality Assessor tool covers the second ML business objective of the project. It contains links to the original raw datasets from which the user can download lemon images and clean them via the Lemon Image Cleaner side tool (which is discussed below). They can then subsequently upload these images to receive a class prediction output made via the model.
![Dataset links and upload option](https://i.imgur.com/gjVGaU8.png)
* Below are examples of the outputs received when uploading a file; namely a binary class prediction along with graphical representation of percentages and an option to download the output DataFrame as a timestamped CSV file.
![Example of processing of uploaded file](https://i.imgur.com/xCj9Amg.png)
![Graphical representation of prediction](https://i.imgur.com/yXKLZFL.png)

### Lemon Image Cleaner tool

* This side tool was incorporated into this project due to a dependency conflict discussed in the Bugs section of this README. It allows the user to upload raw data and outputs a cleaned image ready for model prediction.
![Lemon image cleaner home page and upload](https://i.imgur.com/wooRG4B.png)
![Lemon image upload outputs](https://i.imgur.com/z9rZPrg.png)

### Project Hypothesis 

* This app page displays textual documentation of the projct's hypotheses and analysis of the results and how they align with said hypotheses.
![Project hypothesis section](https://i.imgur.com/iQBsGhw.png)

### Machine learning performance evaluation

* This page documents the dataset size and label frequencies (displaying the target inequality that is later handled by Scikit-learn's class weights function), along with history and evaluation of the project's machine learning model.
![Label frequencies charts](https://i.imgur.com/lisrpIT.png)
* The history of the model by epoch, specifically the validation loss and accuracy per epoch, is displayed in these paired graphs.
![Model history plots](https://i.imgur.com/O2QJPRW.png)
* A confusion matrix for predicted and actual outcomes for the test set is shown here.
![Test set confusion matrix](https://i.imgur.com/zf0CaYm.png)

## Unfixed Bugs
* There are currently no unfixed bugs in this project.

## Deployment
### Heroku

* The App live link is: https://lemon-quality-control.herokuapp.com/ 
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App with desired name
2. Log into Heroku CLI in IDE workspace terminal using the bash command: *heroku login -i* and enter user credentials
3. Run the command *git init* to re-initialise the Git repository
4. Run the command *heroku git:remote -a "YOUR_APP_NAME"* to connect the workspace and your previously created Heroku app.
5. Set the app's stack to heroku-20 using the bash command: *heroku stack:set heroku-20* for compatibility with the Python 3.8.14 version used for this project
6. Deploy the application to Heroku using the following bash command: *git push heroku main*

### Forking the GitHub Project
To make a copy of the GitHub repository to use on your own account, one can fork the repository by doing as follows:

* On the page for the [repository](https://github.com/Dante-Cadiz/lemon-qualitycontrol), go to the 'Fork' button on the top right of the page, and click it to create a copy of the repository which should then be on your own GitHub account.

### Making a Local Clone

* On the page for the [repository](https://github.com/Dante-Cadiz/lemon-qualitycontrol), click the 'Code' button
* To clone the repository using HTTPS, copy the HTTPS URL provided there
* Open your CLI application of choice and change the current working directory to the location where you want the cloned directory to be made.
* Type git clone, and then paste the previously copied URL to create the clone

## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide example(s) on how you used these libraries.

- [NumPy](https://numpy.org/) - Processing of images via conversion to NumPy arrays. Many other libraries used in this project are also dependent on NumPy
- [Pandas](https://pandas.pydata.org/) - Conversion of numerical data into DataFrames to facilitate functional operations
- [Matplotlib](https://matplotlib.org/) - Reading, processing, and displaying image data, producing graphs of tabular data
- [Seaborn](https://seaborn.pydata.org/) - Data visualisation and presentation, such as the confusion matrix heatmap and image dimensions scatter plot.
- [Plotly](https://plotly.com/python/) - Graphical visualisation of data, used in particular on dashboard for interactive charts
- [TensorFlow](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf) - Machine learning library used to build model
- [Keras Tuner](https://keras.io/keras_tuner/) - Tuning of hyperparameters to find best combination for model accuracy
- [Pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) - Reading of annotations of COCO dataset and sorting images by annotations
- [Scikit-learn](https://scikit-learn.org/) - Calculating class weights to handle target imbalance and generating classification report
- [Scikit-image](https://scikit-image.org/) - Application of Sobel filters to image averages to detect edges and dominant features
- [Rembg](https://github.com/danielgatis/rembg) - Used in data cleaning for removal of background elements for raw image data
- [PIL Image](https://pillow.readthedocs.io/en/stable/reference/Image.html) - Used in data cleaning to get bounding boxes of foreground image elements and crop images to thes bounding boxes

## Other technologies used
- [Streamlit](https://streamlit.io/) - Development of dashboard for presentation of data and project delivery
- [Heroku](https://www.heroku.com/) - Deployment of dashboard as web application
- [Git/GitHub](https://github.com/) - Version control and storage of source code
- [Gitpod](https://www.gitpod.io/) - IDE Workspace in which application was developed

## Testing

### Manual Testing

Testing User Stories
As a client, I can navigate easily around an interactive dashboard so that I can view and understand the data presented.

| Feature    | Action     | Expected Result  | Actual Result |
| ----------- | ----------- | ----------- | ----------- |
| Navigation bar | Selecting buttons from main navigation bar | Selected page displayed with correct images/information| Functions as expected |

As a client, I can view and toggle visual graphs of average images (and avergae image difference) and image variabilities for both good and bad quality lemons so that I can observe the difference and understand the visual markers that indicate lemon quality better.

| Feature    | Action     | Expected Result  | Actual Result |
| ----------- | ----------- | ----------- | ----------- |
| Average and standard deviation images button on Lemons Visualiser page | Selecting average image/image variability button | Relevant image plots are rendered | Functions as expected |
| Difference between average image button on Lemons Visualiser page | Selecting difference in average images button | Relevant image plots are rendered | Functions as expected |

As a client, I can view an image montage of either good or bad quality lemons so that I can visually differentiate them.

| Feature    | Action     | Expected Result  | Actual Result |
| ----------- | ----------- | ----------- | ----------- |
| Image montage button on Lemons Visualiser page| Selecting Image Montage button | Select dropdown to choose label for image montage is rendered along with button to create image montage | Functions as expected|
|Image montage creation button on Lemons Visualiser page|Selecting label to display then pressing 'Create Montage' button|Relevant image montage of correct label is displayed|Functions as expected|

As a client, I can access and use a machine learning model so that I can obtain a class prediction on a lemon image provided.
| Feature    | Action     | Expected Result  | Actual Result |
| ----------- | ----------- | ----------- | ----------- |
| ML Performance Metrics Page | Opening ML Performance Metrics page via navbar | Graphs containing label frequencies, model history, and confusion matrix are correctly loaded      | Functions as expected |

As a client, I can provide new raw data of a lemon and clean it so that I can run the provided model on it.

| Feature    | Action     | Expected Result  | Actual Result |
| ----------- | ----------- | ----------- | ----------- |
| Lemon Image Cleaner side application | Uploading raw image data via file upload button | Page outputs rendering of original image and cleaned image below along with correct dimensions listed | Functions as expected |

As a client, I can feed cleaned data to the dashboard to allow the model to predict on it so that I can instantly discover whether a given lemon is of good or poor quality.

| Feature    | Action     | Expected Result  | Actual Result |
| ----------- | ----------- | ----------- | ----------- |
| File uploader on Lemon Quality Assessor page | Uploading cleaned image data via file search button | Page outputs prediction of image class label along with graphical display of class probabilities | Functions as expected |

As a client, I can save model predictions in a timestamped CSV file so that I can keep an account of the predictions that have been made.

| Feature    | Action     | Expected Result  | Actual Result |
| ----------- | ----------- | ----------- | ----------- |
| Download Report button on bottom of Lemon Quality Assessor Page | Pressing download button | File with .csv extension containing timestamped prediction report on uploaded data is saved to user machine | Functions as expected               |

### Validation
- All of the Python code in this project was validated as conforming to PEP8 standards via installation of the pycodestyle package in the workspace terminal via *pip install pycodestyle*
- For the cells of the Jupyter notebooks, an additional step was required of temporarily adding two code cells atop each notebook containing the following code: *%load_ext pycodestyle_magic* and *%pycodestyle_on* .
- I subsequently reran the cells and edited them according to the errors that were documented there.
- For the Streamlit app pages and source code files, I simply edited the code until no errors were recorded in the Problems section of the GitPod workspace.

### Automated Unit Tests
- This project does not currently contain automated unit testing. In future, these will be implemented.

## Bugs

- This project had a major dependency conflict in which the Rembg package required to remove image backgrounds and isolate foreground elements had two key conflicts.
- The first conflict was with TensorFlow version 2.6.0; Rembg required Numpy 1.21.6 while TensorFlow 2.6.0 required a 1.19 version of Numpy. The second conflict was with Streamlit 0.85.0, as Rembg required Click 8.1.3 while Streamlit only supported versions of Click from 7.0 to 8.0. 
- The production solution to these conflicts was to separate the image cleaning process for user submitted images that involved Rembg into a side helper application; [Lemon Image Cleaner](https://lemon-image-cleaner.herokuapp.com/). This was deployed on a later version of Streamlit compatible with the version of Click required for Rembg, along with using the requisite Numpy 1.21.6 version. The user can open this helper application which performs data cleaning tasks on their raw data, then save the images and upload them to the main application for a model prediction.
- Two associated bugs in production were found after the initial deployment on Heroku where creating an image montage fails and throws a Syntax Error 'Not a PNG file', and .h5 file for model fails to load. 
- These were found to be because the input image files and .h5 model files were stored in Git LFS which Heroku does not support.
- To fix this bug, I moved the files previously stored in LFS out of LFS and into regular Git storage.
- Upon redeployment with the files relocated, deployment was initially rejected due to the compressed slug size being larger than Heroku's 500MB limit.
- This was fixed by creating a .slugignore file that allowed Heroku to ignore most input images, as such only pushing a selection of the validation set input images to Heroku - still enough to create the image montage.


## Credits 


### Content 

- The two datasets for this project were found on Kaggle.
- [Dataset 1](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset) is under CCO Public Domain License.
- [Dataset 2](https://www.kaggle.com/datasets/maciejadamiak/lemons-quality-control-dataset) is copyrighted by SoftwareMill under MIT License and general use of it such as this project is permitted. For more infomation about this copyright, refer to the README of the dataset.
- Many of the project's functions were transferred from Code Institute's sample Malaria Detector project, along with inspiration for the workflow direction
- Code for calculating class weights came from [this StackOverflow answer](https://stackoverflow.com/questions/42586475/is-it-possible-to-automatically-infer-the-class-weight-from-flow-from-directory/67678399#67678399)
- Information about typical lemon defects and testing methods/metrics during quality control came from [Clarifruit](https://www.clarifruit.com/knowledge-base/fresh-produce-categories/lemons/) and the UK Government's [Specific Marketing Standards](https://www.gov.uk/guidance/comply-with-marketing-standards-for-fresh-fruit-and-vegetables)
- The workflow for hyperparameter tuning was adapted from TensorFlow's [Keras Tuner Tutorial](https://www.tensorflow.org/tutorials/keras/keras_tuner)

## Acknowledgements 
* My mentor, Mo Shami, for supervising this project
* The #project-portfolio-5-predictive-analytics channel on Slack and especially the Code Institute Tutors who participate there for providing technical advice

