## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/alexiaxreyes/SatelliteCoastlineNetwork/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

![](http://serc.carleton.edu/images/NAGTWorkshops/oceanography/big_sur_coastline.jpg)

## Satellite Coastline Network 
Description: Application for a CNN-Supervised Classification of remotely sensed imagery with deep learning. 

### Dependencies 
* Keras (Tensorflow-GPU v1.14 as the backend) 
* Sciki-Learn 
* Imbalanced-Learn tool box
* Scikit-Image 
* Pandas 

### Basic Installation 
After installing dependencies, the application can be tested with the instructions, data and base model provided. 

### Data Preparation 
![](https://www.spaceflightinsider.com/wp-content/uploads/2016/07/worldview-3-1.jpg)
In order to generate coastline extraction results with higher resolution, [WorldView-2 Satellite](https://www.satimagingcorp.com/satellite-sensors/worldview-2/) image products were used to identify the boundary between land and open water for Kaktovik and Wainwright, Alaska. We recommend that the data be structured as: CoastlineName_Number.jpg. The number must be at least 4 digits (CoastlineName_0022.jpg), but can be more if nessesary (exampe 5-digit, CoastlineName_12345.jpg). The associated classification is expected to have the same filename but with a prefix of 'SCLS_' and a tif format (SCLS_CoastlineName_0022.tif). The default number of classes in the code and in the label data found on the repository is 2: water and land. Users can alter the number of classes for other studies as needed. However, all the code and models function by tiling the input imagery in sub-images of 50x50 pixels.

### Convolutional Neural Network (CNN) Training 
After data preparation, the script TrainCNN.py can be used to train the Keras H5 base model architecture with pretrained weights as downloaded. User options are at the start. Elements marked 'Path' or 'Empty' need to be edited. It is recommended to set the ModelTuning variable to True and run the tuning procedure for the CNN. This will output a figure and the correct number of tuning epochs can be set as the point where the loss and accuracy of the validation data begin to diverge from the loss and accuracy of the training data. Once this is established, the script must be run again with ModelTuning set to False and the correct value for Tuning. This will save the Keras model with a .h5 extension and it will also save a class key as a small csv file. Once these options are edited in the code no switches are required. 
```Python
Train CNN
```

### Convolutional Neural Network (CNN)- Supervised Classification (CSC) Execution 
After development of trained CNN model, CSC performance can be evaluated with CnnSupervisedClassification.py. The images to test must follow the same naming convention and all have an existing set of manual labels as used in the CNN training phase above. Again variables currently set to 'Path' or 'Empty' must be edited in the code. The CSC is currently set to use a Multilayer Perceptron (MLP) to perform the phase 2, pixel-level, classification. In this phase, the CNN classification output for a specific image will be used as training data for that specific image. The script will execute and output performance metrics for each image. csv files with a CNN_ prefix give performance metrics for the CNN model with F1 scores and support (# of pixels) for each class. MLP_ files give the same metrics for the final CSC result after the application of the MLP. A 4-part figure will also be output showing the original image, the existing class labels, the CNN classification and the final CSC classification labelled either MLP. Optionally, a saved class raster can also be saved to disk for each processed image.
```Python
CnnSupervisedClassification
```
