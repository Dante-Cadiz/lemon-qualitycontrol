## Dataset Content
* This project uses 2 datasets from separate sources that are later combined and analysed simultaneously to fulfil the project's business objectives.
* The first dataset for this project is a large collection of 2D images of lemons in a zipped file, pre-labelled as either being of good or poor quality based on the visual condition of their rind. There is also a third category of images; blank backgrounds not containing a lemon. I have chosen to ignore this third category here as analysing it is not necessary to fulfil the project's business requirements. These images are all 300x300px but will be resized during the image preparation process to allow for more time-efficient model learning. This dataset was found here on [Kaggle](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset)
* The second dataset is another large collection of 2D images, annotated manually by the data owner using OpenCV's CVAT. The annotations capture typical lemon quality issues such as illness, gangrene, mould, visual blemishes, and dark style remains. The annotations also cover structural regions of the lemon and photographic features such as lighting issues or blur. The annotated dataset can be accessed, read and viewed via pycocotools' COCO function. These images are all 1056x1056px but will be resized during the image preparation process to allow for more time-efficient model learning. This dataset was found here on [Kaggle](https://www.kaggle.com/datasets/maciejadamiak/lemons-quality-control-dataset)

## Business Requirements
* The stakeholder for this project is the owner of a small-scale commerical fruit preparation factory that supplies local markets. 
* The factory handles and processes fruit delivered from local farms to ensure it is ready for supermarket display and sale. 
Heuristic
- The factory's current process for separating its lemons into those suitable and unsuitable for retail currently involves: 
Conveyor belt machinery that measures diameter
Performing chemical tests such as ripening gas concentration and assessment for common types of rot
Factory employees manually inspecting and separating the fruits based on a visual assessment of blemishes and overall skin quality. 
- The factory owner wants to eliminate this final process and replace it with a computerised image assessment, speeding the overall preparation of lemons up and removing reliance on individual human judgement and susceptibility to error. 

- As such, the project has the following business objectives:
- The client is interested in analysing the visual difference between good and poor quality lemons, specifically the visual markers that define a poor quality lemon. This will be satisfied via conventional data analysis: finding the average image and variability per label in the data set, as well as the contrast between said labels.
- The client is interested in accurately and instantly predicting from a given image whether a lemon is of good or poor quality. This business objective will be delivered by the development and deployment of a TensorFlow deep learning pipeline 
- The content satisfying these two business objectives will be combined into a Streamlit dashboard and delivered to stakeholders.

## Agile Development / User Stories
- How are they fulfilled via dashboard

## Hypothesis and how to validate?
* There will be a clear visual difference noticeable between the average images for each label; good and poor quality lemons
* Put info about typical lemon diseases/defects here
* Validation accuracy


## Rationale to map the business requirements to the Data Visualizations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualizations and ML tasks
- The client is interested in defining and analysing the visual difference between good and poor quality lemons. This will be satisfied via conventional data analysis: finding the average image and variability per label in the data set, as well as the contrast between said labels.

## ML Business Case
* In the previous bullet, you potentially visualized a ML task to answer a business requirement. You should frame the business case using the method we covered in the course 
- The client is interested in accurately predict from a given image whether a lemon is of good or poor quality. This business objective will be delivered by the development and deployment of a TensorFlow deep learning pipeline 
- The eventual target of the machine learning pipeline is a binary classification label; good or poor quality.
- The performance and quality of the model
 and its ability to identify and discern visual features that mark a bad quality lemon
- The stakeholder has decided that the most important metric for evaluating the success of the machine learning model is recall on the negative results of a poor quality lemon. This is because providing possibly poor quality lemons to supermarkets is a worse outcome than mistakenly discarding good quality ones. A preliminary recall score threshold to assess the model's success has been set at 0.98.

- Accuracy and F1 score will also be taken into consideration - with 95% accuracy as minimum threshold
- Output
- Heuristic

## Hypotheses and how to Validate

1.
2.
3.


## Dashboard Design
* List all dashboard pages and its content, either block of information or widgets, like: buttons, checkbox, image, or any other item that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a give feature (for example, in the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly in case all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide example(s) on how you used these libraries.

- Numpy
- Pandas
- Matplotlib
- Seaborn - data visualisation
- Plotly - data visualisation
- Tensorflow - image preprocessing, model building, hyperparameter tuning
- Scikit-image - image processing/transformation
- Rembg - data cleaning

## Testing

## Bugs

- Two associated bugs in production on Heroku where creating an image montage fails and throws a Syntax Error 'Not a PNG file', and .h5 file for model fails to load. 
- These were found to be because the input image files and .h5 model files were stored in Git LFS which Heroku does not support.
- To fix this bug, I moved the files previously stored in LFS out of LFS and into regular Git storage.

## Credits 

* In this section you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- Much of the project's functions came from Code Institute's sample Malaria Detector project, along with inspiration for the workflow direction
- Code for calculating class weights came from [this StackOverflow answer](https://stackoverflow.com/questions/42586475/is-it-possible-to-automatically-infer-the-class-weight-from-flow-from-directory/67678399#67678399)
- Information about typical lemon defects and testing methods during quality control came from [Clarifruit](https://www.clarifruit.com/knowledge-base/fresh-produce-categories/lemons/)
- The workflow for hyperparameter tuning was adapted from

### Media





## Acknowledgements (optional)
* In case you would like to thank the people that provided support through this project.

