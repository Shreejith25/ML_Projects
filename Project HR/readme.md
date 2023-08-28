Analysis and Prediction of Employee Promotions Using Machine Learning
HR dataset that can be used to predict employee promotions within an organization based on various factors/features such as employee ID, department, region, education, etc. The dataset contains (54809 observations and 14 Features) historical employee records and promotion outcomes, offering a valuable resource for understanding the factors that influence career advancement. The goal of this dataset analysis is to develop a predictive model using ML techniques to determine the likelihood of an employee being promoted and deploy a web app for the same using Flask

Problem Statement
This is a company’s HR dataset. Employees are promoted within the organization on an annual basis. Therefore, using the dataset, we must determine whether an employee has been promoted or not

Features in data set
employee_id: Unique ID for employee
department: Department of employee
region: Region of employment (unordered)
education: Education Level
gender: Gender of Employee
recruitment_channel: Channel of recruitment for employee
no_ of_ training: no of other training completed in the previous year on soft skills, technical skills, etc.
age: Age of Employee
previous_ year_ rating: Employee Rating for the previous year
length_ of_ service: Length of service in years
awards_ won?: if awards won during the previous year then 1 else 0
avg_ training_ score: Average score in current training evaluations
is_promoted: (Target) Recommended for promotion

HR Promotion Prediction
This repository contains a Python script for predicting employee promotions based on a provided HR dataset. The script performs data preprocessing, feature engineering, and uses an XGBoost classifier for the prediction. Below is a step-by-step explanation of the code:

Table of Contents
Introduction
[Exploratory Data Analysis](#Exploratory Data Analysis)
Data Preprocessing
Feature Engineering
Data Splitting
Balancing the Dataset
Model Selection
Model Building
Introduction
This script aims to predict employee promotions based on a provided HR dataset. It uses machine learning techniques, specifically XGBoost, to make predictions.

Getting Started
To run this code, you'll need to have the following Python libraries installed:

numpy
pandas
matplotlib & seaborn
scikit-learn
imbalanced-learn
xgboost
Data Preprocessing
The script begins by importing necessary libraries and loading the HR dataset from the "HR dataset Project.csv" file. It then performs the following data preprocessing steps:

Handling Missing Education Values: Null values in the "education" column are replaced with the mode of education for individuals with the same age.

Handling Missing Previous Year Rating: Null values in the "previous_year_rating" column are filled with 0, assuming these are new employees with 1 year of service.

Ordinal Encoding for Education: The "education" column is ordinal encoded into numeric values.

One-Hot Encoding for Gender, Recruitment Channel, and Department: These categorical columns are one-hot encoded and concatenated with the original dataset. The original columns are then dropped.

Extracting Region Numbers: Numeric values are extracted from the "region" column.

Feature Scaling: The "avg_training_score" column is scaled using StandardScaler.

Feature Reduction
After preprocessing, the script performs feature engineering by removing the "employee_id" column.

Data Splitting
The dataset is split into training and testing sets. The independent variables (features) are stored in x, and the target variable (promotion status) is stored in y. The script uses scikit-learn's train_test_split function to perform this split with a 75%/25% train/test ratio.

Balancing the Dataset
The training dataset is balanced using the SMOTETomek technique from the imbalanced-learn library. This helps address the class imbalance problem often present in promotion prediction datasets.

Model Selection

Model Choice
We have chosen to use the XGBoost classifier for this task. XGBoost is an ensemble machine learning algorithm known for its high performance and flexibility in handling structured data. It is particularly suitable for classification problems like the one we are dealing with, which involves predicting whether an employee will be promoted or not.

Model Hyperparameters
The XGBoost classifier is configured with the following hyperparameters:

Booster: 'gbtree'
Evaluation Metric: 'auc' (Area Under the Receiver Operating Characteristic Curve)
Learning Rate: 0.2
Max Depth: 270
Min Child Weight: 3
Number of Estimators (Trees): 1127
Objective: 'binary:logistic'
Scale Pos Weight: 2
Tree Method: 'auto'
These hyperparameters were selected through a combination of manual tuning and hyperparameter optimization techniques to achieve the best model performance.

Model Training
The training of the XGBoost classifier is done using the balanced training dataset obtained after applying the SMOTETomek technique. Balancing the dataset is crucial to prevent model bias towards the majority class, which is a common issue in promotion prediction tasks where the number of promoted employees is significantly lower than non-promoted ones.

Model Evaluation
To assess the model's performance, various evaluation metrics such as accuracy, precision, recall, F1-score, and the ROC-AUC score are calculated on the test dataset. These metrics help us understand how well the model is performing in terms of predicting promotions accurately and efficiently.

Model Deployment
After training and evaluating the model, it can be deployed in a production environment to make real-time predictions on new data. The deployment process may involve saving the trained model to a file for easy retrieval and integration into other applications.

This section provides an overview of the model selection process, hyperparameters, training, evaluation, and deployment considerations for predicting employee promotions using the XGBoost classifier.

Model Building
Finally, the script builds an XGBoost classifier with specified hyperparameters for promotion prediction. The trained model is stored in the variable model and can be used for making predictions on new data.

HR Promotion Prediction Web Application
This repository contains a Flask-based web application for predicting employee promotions based on a trained machine learning model. Users can input employee details, and the application will predict whether the employee will be promoted or not.

Table of Contents
Introduction
Getting Started
Web Application
Usage
Dependencies
Introduction
In the modern HR landscape, it's essential to identify employees who have the potential to be promoted within an organization. This web application leverages a machine learning model to automate this prediction process based on various employee attributes.

Getting Started
To get started with this web application, follow the steps below:

Clone this repository to your local machine:

git clone https://github.com/Shreejith25/ML_Projects/tree/main/Project%20HR
Navigate to the project directory:

cd hr-promotion-predictor
Install the required Python libraries (make sure you have Python 3.x installed):

pip install flask numpy scikit-learn
Run the Flask application:

python app.py
The application will start running on http://localhost:9000. You can access it through your web browser.

Web Application
The web application consists of two main pages:

Home Page (index.html): This is the landing page where users can input employee details to get a promotion prediction.

Result Page (result.html): This page displays the prediction result, indicating whether the employee is predicted to be promoted or not.

Usage
Access the application by opening your web browser and navigating to http://localhost:9000.

On the Home Page, enter the required employee details, including:

Employee ID
Age
Education
Gender
Department
Region
Recruitment Channel
Training
Previous Year Rating
Length of Service
Average Training Score
KPI (Key Performance Indicator)
Awards Won
Click the "Predict" button to submit the information.

The application will process the data using the trained machine learning model and display the prediction result on the Result Page.

Dependencies
The web application relies on the following Python libraries:

Flask: Used for creating the web application.
NumPy: Required for numerical operations.
scikit-learn: Used for preprocessing data and making predictions.
Ensure you have these libraries installed as mentioned in the "Getting Started" section.

This web application provides an intuitive interface for predicting employee promotions based on the provided employee details. It streamlines the decision-making process in HR and helps identify employees with high promotion potential.

Feel free to modify this script as needed for your specific use case and dataset.
