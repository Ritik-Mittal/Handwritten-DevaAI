# Devnagari Handwritten Character Recognition

This repository contains a project for recognizing handwritten Devnagari characters using Principal Component Analysis (PCA) for dimensionality reduction and Gaussian Naive Bayes (GaussianNB) for classification.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Feature Extraction](#feature-extraction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to perform handwritten character recognition using the Devnagari Handwritten Character Dataset. It involves various steps, including data preprocessing, dimensionality reduction using PCA, and classification using the Gaussian Naive Bayes algorithm.

## Dataset

The dataset contains handwritten characters categorized into 46 classes. Each class represents a Devnagari character. The dataset is divided into training and testing subsets.

## Installation

To run this project, you need to have Python and the following packages installed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Data Preparation

1. **Download the dataset** from the specified source:
    ```bash
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00389/DevanagariHandwrittenCharacterDataset.zip
    unzip DevanagariHandwrittenCharacterDataset.zip
    ```

2. **Run the data preparation code** to preprocess the data and create the necessary training and testing sets.

```python
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Define functions for data preprocessing
def image_flattening(img_np_array):
    return img_np_array.reshape(img_np_array.shape[0] * img_np_array.shape[1],)

def read_image(image_path):
    return plt.imread(image_path)

def single_dir_img_path_list(base_train_dir_path):
    img_list = os.listdir(base_train_dir_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        img_path_list = list(pool.map(lambda x: base_train_dir_path+'/'+x, img_list))
    return img_path_list

def single_category_dir_path(base_train_path):
    dirs = os.listdir(base_train_path)
    dir_path_list = []
    for dir_name in dirs:
        dir_path_list.append(base_train_path + '/' + dir_name)
    return dir_path_list

base_train_path = "/content/DevanagariHandwrittenCharacterDataset/Train"
base_test_path = "/content/DevanagariHandwrittenCharacterDataset/Test"

dir_path_list = single_category_dir_path(base_train_path)
dir_test_path_list = single_category_dir_path(base_test_path)

def single_dir_imgs_np_array_list(img_path_list):
    with ThreadPoolExecutor(max_workers=4) as pool:
        imgs_np_array_list = list(pool.map(read_image,img_path_list))
    return imgs_np_array_list

def single_dir_imgs_matrix(imgs_np_array_list):
    with ThreadPoolExecutor(max_workers=4) as pool:
        reshaped_imgs_list = list(pool.map(image_flattening, imgs_np_array_list))
    return np.array(reshaped_imgs_list)

def generate_feature_matrix(dir_path_list):
    reshaped_imgs_matrix_list = []
    for single_dir_path in dir_path_list:
        img_path_list = single_dir_img_path_list(single_dir_path)
        imgs_np_array_list = single_dir_imgs_np_array_list(img_path_list)
        reshaped_imgs_matrix = single_dir_imgs_matrix(imgs_np_array_list)
        reshaped_imgs_matrix_list.append(reshaped_imgs_matrix)
    return reshaped_imgs_matrix_list

reshaped_imgs_matrix_list = generate_feature_matrix(dir_path_list)
reshaped_test_imgs_matrix_list = generate_feature_matrix(dir_test_path_list)

training_data_input_features = np.concatenate(reshaped_imgs_matrix_list)
testing_data_input_features = np.concatenate(reshaped_test_imgs_matrix_list)

print(f'Shape of Training Data: {training_data_input_features.shape}')
print(f'Shape of Training Data: {testing_data_input_features.shape}')

df_train = pd.DataFrame(training_data_input_features, columns=["Pixel_" + str(i) for i in range(1, training_data_input_features.shape[1] + 1)])
df_test = pd.DataFrame(testing_data_input_features, columns=["Pixel_"+str(i) for i in range(1,testing_data_input_features.shape[1]+1)])

def Label_maker(df, img_path_list):
    Image_labels = []
    for dir_name in os.listdir(base_train_path):
        Image_labels.extend([dir_name]*len(img_path_list))
    df['Image_Labels'] = Image_labels
    return df

df_train = Label_maker(df_train, image_path_list)
df_test = Label_maker(df_test, test_img_path_list)

df_train.head()
df_test.head()
```

## Feature Extraction

To feed the data into a machine learning model, we need to perform dimensionality reduction. Here, Principal Component Analysis (PCA) is used to reduce the number of features while retaining as much variance as possible.

```python
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

training_features = df_train.iloc[:,:-1]
training_labels = df_train.iloc[:,-1]
testing_features = df_test.iloc[:,:-1]
testing_labels = df_test.iloc[:,-1]

def fetching_performance_metrics(num_components):
    pca = PCA(n_components=num_components)
    reduced_training_features = pca.fit_transform(training_features)
    reduced_testing_features = pca.transform(testing_features)

    obj = GaussianNB()
    obj.fit(reduced_training_features, training_labels)

    y_pred = obj.predict(reduced_training_features)
    train_metrics = get_metrics(training_labels, y_pred)

    test_pred = obj.predict(reduced_testing_features)
    test_metrics = get_metrics(testing_labels, test_pred)

    CR = {'Train': train_metrics, 'Test': test_metrics}
    return CR

def get_metrics(true_labels, predicted_labels):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true=true_labels, y_pred=predicted_labels)
    metrics['Precision'] = precision_score(y_true=true_labels, y_pred=predicted_labels, average='macro')
    metrics['Recall'] = recall_score(y_true=true_labels, y_pred=predicted_labels, average='macro')
    metrics['F1_Score'] = f1_score(y_true=true_labels, y_pred=predicted_labels, average='macro')
    return metrics

num_pixels = np.arange(100,1024,50)

complete_report = dict()
for num_components in num_pixels:
    complete_report[num_components] = fetching_performance_metrics(num_components)

complete_report
```

## Model Training and Evaluation

The model is trained using the training data and validated on the test set. The evaluation metrics include accuracy, precision, recall, and F1-score.

## Results

The model's performance metrics for various numbers of principal components are stored in the `complete_report` dictionary.

## Usage

To use the trained model for inference, run the following code:

```python
import numpy as np

# Load and preprocess the image
image = ...  # Load your image here
image = image.reshape(1, -1)  # Reshape if necessary

# Apply PCA transformation
image_reduced = pca.transform(image)

# Predict the class
prediction = model.predict(image_reduced)
print(f'Predicted class: {prediction[0]}')
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.
