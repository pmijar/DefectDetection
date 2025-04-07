## DefectDetection

### Overview  
This program is designed to spot defects in steel sheets using deep learning techniques. We’re working with a training dataset that includes 7,095 images, and we’re applying data augmentation to boost both the size and variety of this dataset. The project involves comparing different deep learning architectures, and we’ll be fine-tuning the selected model to enhance its performance.

### Contents  
1. Objective  
2. Dataset  
3. Data Augmentation  
4. Model Comparison  
5. Fine-tuning   
6. Results  

### Objective  
The main aim of this project is to create a deep learning model that can accurately identify defects in steel sheets. We’ll train the model using a relatively small dataset of 7,095 labeled images, employing techniques like data augmentation, model comparison, and fine-tuning to improve its accuracy.

### Dataset  
Our dataset is made up of 7,095 images of steel sheets, each labeled with the specific type of defect. We’ll divide the dataset into training and validation sets. To prepare the images, we normalize the dataset by scaling the integer values down (by dividing by 255).  
Source: https://www.kaggle.com/competitions/severstal-steel-defect-detection/data

### Data Augmentation
Due to the limited size of the dataset, data augmentation is applied to increase the size and diversity of the training data. Data augmentation applied are as follows:

*	rescale=1./255: This step normalizes the pixel values by scaling them down to a range between 0 and 1.
*	shear_range=0.2: It applies random shear transformations to the images, allowing for variations within a 0.2 range. 
*   zoom_range=0.2: This feature randomly zooms in or out on the images, giving a variation of up to 20%.*  
*   horizontal_flip=True: It randomly flips images horizontally to enhance augmentation. 
*   vertical_flip=True: This option randomly flips images vertically for added variety. 
*   brightness_range=[0.8, 1.2]: It adjusts the brightness of the images randomly, ranging from 80% to 120% of the original brightness.

### Model Comparison
The program compares multiple deep learning models to find the one that best performs on the steel defect detection task. Models tested include:
 * InceptionV3
 * EfficientNetB0
 * Xception

Each model is trained and evaluated using the same dataset, and accuracy performance metrics were used for comparison across each of the selected models.

#### Fine-tuning
Applied the fine-tuning to improve the model's performance further. Fine-tuning consists of:
 * Unfreezing the deep learning models from layer = 50
 * Adjusting the learning rate = 0.0001
 * Early stopping

### Results
Of the three training models that we identified for training and fine-tuning, we found that deep learning Xception performed that best. 
