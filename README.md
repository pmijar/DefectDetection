## DefectDetection

### Overview  
This program is designed to spot defects in steel sheets using deep learning techniques.This is a classification problem by trying to find the class[1, 2, 3, 4] of defect based on the image captured. Working with a training dataset that includes 7,095 images, and applying data augmentation to boost both the size and variety of this dataset. 
The project involves comparing different deep learning models, and fine-tuning these selected models to enhance its performance and improve defect classification.

### Contents  
1. Objective  
2. Dataset  
3. Data Augmentation  
4. Model Comparison  
5. Fine-tuning   
6. Results  

### Objective  
The main aim of this project is to create a deep learning model that can accurately identify defects in steel sheets. Models will be using a relatively small available dataset of 7,095 labeled images,we will be employing techniques like data augmentation, model comparison, and fine-tuning to improve its accuracy.

### Dataset  
Our dataset is made up of 7,095 images of steel sheets, each labeled with the specific type of defect, the defect class range from 1 through 4. The dataset will be divived into training and validation sets. During data prep, the image pixel value of dataset is normalized (by dividing by 255). 

Dataset Source: https://www.kaggle.com/competitions/severstal-steel-defect-detection/data
_Notes: Please note that only limited image files were uploaded in the data, Use the kaggle data set for the entire data to run against the attached jupyter notebook_

### Data Augmentation
Due to the limited size of the dataset, data augmentation is applied to increase the size and diversity of the training data. Data augmentation applied are as follows:

*	__rescale=1./255:__ This step normalizes the pixel values by scaling them down to a range between 0 and 1.
*	__shear_range=0.2:__ It applies random shear transformations to the images, allowing for variations within a 0.2 range. 
*   __zoom_range=0.2:__ This feature randomly zooms in or out on the images, giving a variation of up to 20%.*  
*   __horizontal_flip=True:__ It randomly flips images horizontally to enhance augmentation. 
*   __vertical_flip=True:__ This option randomly flips images vertically for added variety. 
*   __brightness_range=[0.8, 1.2]:__ It adjusts the brightness of the images randomly, ranging from 80% to 120% of the original brightness.

### Model Comparison
The program compares multiple deep learning models to find the one that best performs on the steel defect detection task. Models tested include:
 * __InceptionV3__
 * __EfficientNetB0__
 * __Xception__

Additioanlly, The top custom layers added to the pre-trained model are as follows:

#### Layer-1 
* __Conv2D Layer:__ We start with a 2D convolutional layer featuring 32 filters and a kernel size of 3, which processes the output from the base model. Next, we have:
* __BatchNormalization:__ This step normalizes the activations from the previous layer.
* __Activation ("ReLU"):__ Here, we apply the ReLU activation function to introduce non-linearity.
* __MaxPooling2D:__ A max pooling operation is performed with a pool size of 3 and a stride of 2, which helps to downsample the feature maps.

#### Layer-2
* __Conv2D Layer:__ Another convolutional layer with 64 filters and a smaller kernel size of 2, followed by:
* __BatchNormalization__
* __Activation ("ReLU")__
* __MaxPooling2D:__ This layer uses a pool size of 2 and a stride of 1 to further downsample the feature maps.
* __Dropout Layer:__ To reduce overfitting, added a dropout layer with a rate of 0.2, which randomly sets 20% of the activations to zero during training.
* __BatchNormalization:__ Additional normalization after the dropout layer.
* __Flatten Layer:__ This layer flattens the feature maps into a one-dimensional vector, feeding the fully connected Dense layer.

#### Dense Layer: There are two fully connected (dense) layers:
* The first dense layer consists of 256 neurons with ReLU activation.
* The second dense layer has 128 neurons, also with ReLU activation.

#### Output Layer: 
* The last dense layer generates the output, corresponding to the number of defect classes, utilizing the softmax activation function to classify into multiple categories.

Each of these models are trained and evaluated using the same dataset, with accuracy performance metrics used for comparison across each of the selected models.

 Model training accuracy and loss Stats:
```
 Results for InceptionV3:
 * Training Accuracy: 0.8211768865585327
 * Validation Accuracy: 0.816067636013031
 * Training Loss: 0.42351657152175903
 * Validation Loss: 0.4630342125892639


Results for EfficientNetB0:
 * Training Accuracy: 0.724277675151825
 * Validation Accuracy: 0.7322058081626892
 * Training Loss: 0.8725362420082092
 * Validation Loss: 0.85508131980896


Results for Xception:
 * Training Accuracy: 0.8282241225242615
 * Validation Accuracy: 0.8379140496253967
 * Training Loss: 0.3997619152069092
 * Validation Loss: 0.42924144864082336
```
![Training Accuracy](images/TrainingModels_Accuracy.png)
![Training Loss](images/TrainingModels_Loss.png)

#### Fine-tuning
Further applied the fine-tuning to improve the model's performance. Fine-tuning consisted of:
 * Unfreezing the deep learning models from layer = 50
 * Adjusting the learning rate = 0.0001
 * Early stopping

 Model Fine-tuning accuracy and loss Stats

```
 Results for InceptionV3:
 * Fine tuned Training Accuracy: 0.8801973462104797
 * Fine tuned Validation Accuracy: 0.858350932598114
 * Fine tuned Training Loss: 0.29150131344795227
 * Fine tuned Validation Loss: 0.3806813061237335

Results for EfficientNetB0:
 * Fine tuned Training Accuracy: 0.7943974733352661
 * Fine tuned Validation Accuracy: 0.7575757503509521
 * Fine tuned Training Loss: 0.48865455389022827
 * Fine tuned Validation Loss: 0.69825279712677


Results for Xception:
 * Fine tuned Training Accuracy: 0.8904157876968384
 * Fine tuned Validation Accuracy: 0.8618745803833008
 * Fine tuned Training Loss: 0.2580391466617584
 * Fine tuned Validation Loss: 0.3762108385562897
 ```
 ![Fine-tuned Accuracy](images/FineTunedModels_Accuracy.png)
![Fine-tuned Loss](images/FineTunedModels_Loss.png)

### Results
Of the three pretrained models that we identified for training and fine-tuning, we found that Xception model performed the best with respect to accuracy and loss.
