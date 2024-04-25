
# ESA Satellite Collision Avoidance Challenge


## Introduction
This project implements modeling potential satelite collisions in advance using machine learning.
The dataset is provided from the [Collision Avoidance Challenge  by ESA](https://kelvins.esa.int/collision-avoidance-challenge/home/).
The dataset can be downloaded [here](https://kelvins.esa.int/collision-avoidance-challenge/data/).  
The dataset consists of Conjunction Data Message (CDM) that grouped together can be interpreted as a set of Multivariate Time Series.
  
## Contents

### 1. Installation
### 2. Methodology
### 3. Evaluation
### 4. Future improvements

## Installation and Usage
To set up the project, follow these steps:
1.  Create a virtual environment with python 3.10 and activate it:

> conda create -n vyoma_test python=3.10
> conda activate vyoma_test

2. Install the required dependencies:
> pip install -r requirements.txt

3. All the configurations for running the pipeline can be found and also changed as per user environment in the config.yaml file.
4. Run the pipeline simply by running `python main.py` command.
5. `train_mode = False` skips the training pipeline and runs a pretrained model on the test dataset.

## Methodology
The project follows a standard machine learning workflow:

1.  **EDA and Data Preprocessing**: First, exploratory data analysis was conducted to understand the distribution of data. Extensive data processing was also conducted to transform the data in a usable form. Please refer to the [data_preprocessing.ipynb](/data_pipeline/data_preprocessing.ipynb) notebook for further details.
2.  **Feature Selection**: Multiple feature selection techniques were used to select the most descriptive features from the multivariate time series dataset. Please refer to the [feature_selection.ipynb](/data_pipeline/feature_selection.ipynb) notebook for further details.
3.  **Model Selection**: Due to the sequential nature of the data, the GRU model was selected for this challenge.
4.  **Model Training and Evaluation**: The model was trained to predict the risk value of an event, given the multivariate sequential data of the two orbiting objects: 'chaser' and 'target'. Further details about evaluation metric and results can be found in the Evaluation section.

## Evaluation

 - The evaluation was performed using the custom evaluation metrics provided for this challenge which combined the mean_squared_error of prediction with the f-score of predicting high_risk events.
 -  This evaluation metric was also used as the loss function for training of the GRU model. The exact definiton of the function can be found in the [trainer.py](/trainer/trainer.py) script.
 - It was found after multiple experiments that instead of predicting the accurate risk_value, the model tended to predict every event as a high risk event.
 - The root cause of this behavior can be traced to the way the evaluation metric was defined in the training loop. 
 - A high penalty was applied on missclassification of high risk events.
 - To avoid this high penalty (and further decrease the loss value), the model resolved to predicting every event as a high risk event.
 - An example output from a batch is provided below:
 

    ground_truth_risk: 
    [tensor([[-16.5814, -23.1741, -20.6099, -19.6735, -18.8891, -17.1776, -12.7464,-19.4563, -20.3117, -20.2923, -26.0682, -27.1323]])]
    predicted_risk:
    [tensor([[-5.3616, -5.5601, -5.7264, -5.7105, -5.8139, -5.6024, -5.5572, -5.3110, -5.2599, -5.0870, -4.7612, -4.1504]])]
    
## Future Improvements

  - Weighted Dataset: Due to the imbalance of the risk values, weighted sampling of the dataset could be performed to better capture the high_risk events.
 - Model size: The effect of increased model size should be analysed for the given task.
 - Feature transformations: Scale transformation of features could be further studied to improve model learning.

