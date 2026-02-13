# HOUSE-PRICE-PREDICTION-SYSTEM
House Price Prediction using Machine Learning
Project Overview

This project is a Machine Learning–based house price prediction system focused on residential properties in Chennai.

The model predicts house prices based on key property features such as area, number of bedrooms, bathrooms, floor level, age, furnishing status, and location within Chennai.

The project also includes a visualization that shows which Chennai areas have the highest average house prices.

Objective

The objective of this project is to:

Build a predictive model for house price estimation

Understand the impact of different property features on price

Compare average prices across Chennai locations

Create an interactive system where users can manually input house details and receive a predicted price

Dataset

The dataset contains 3000 records and includes the following features:

area (in square feet)

bedrooms

bathrooms

floor

age (years)

furnishing (0 = Unfurnished, 1 = Semi-Furnished, 2 = Fully Furnished)

location (Chennai locality)

price (target variable in INR)

Locations included:

Anna Nagar

Adyar

Velachery

OMR

Tambaram

Porur

T Nagar

Besant Nagar

Guindy

Chromepet

The dataset simulates realistic Chennai pricing patterns based on locality and property features.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Machine Learning Model

The project uses:

Random Forest Regressor

Why Random Forest?

Captures complex relationships between features

Handles non-linear pricing patterns

Provides high prediction accuracy

Reduces overfitting through ensemble learning

How the Model Predicts Price

The dataset is split into training and testing sets.

Numeric features are scaled using StandardScaler.

Location (categorical feature) is encoded using OneHotEncoder.

Random Forest builds multiple decision trees.

Each tree predicts a price.

Final prediction is the average of all trees.

The model is evaluated using:

R² Score

Mean Squared Error (MSE)

Visualization

The project includes a bar chart showing:

Average house price by Chennai location.

This helps identify:

High-value areas

Mid-range areas

Affordable areas

The visualization allows easy comparison of property prices across different regions.

How to Run the Project

Clone the repository.

Ensure the dataset file is in the same folder as the Python script.

Install required libraries:

pip install pandas numpy scikit-learn matplotlib


Run the program:

python house_price_prediction.py


Enter house details manually when prompted.

The model will output the predicted price.

A visualization of average prices by location will be displayed.

Example Manual Input

Area (sq ft)

Bedrooms

Bathrooms

Floor

Age

Furnishing status

Location

The system then predicts the house price in INR.

Skills Demonstrated

Data preprocessing

Handling categorical variables

Feature scaling

Ensemble machine learning

Model evaluation

Data visualization

Interactive user input handling

Learning Outcome

This project demonstrates the application of machine learning techniques in real estate price prediction and highlights how location and property features influence housing prices in Chennai.
