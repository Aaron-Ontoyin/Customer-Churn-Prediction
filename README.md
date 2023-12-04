# Customer Churn Prediction

![Cover Image](cover-img.png)

## Overview

This project focuses on predicting customer churn using various machine learning models. The project is structured into several modules and a Jupyter Notebook for data investigation and model testing.

The ML models investigated here are:
1. Logistic Regression Model
    - The logistic regression model is a classification algorithm that is used to predict the probability of a binary outcome. 
    - It is commonly used in machine learning for tasks such as customer churn prediction, spam detection, and fraud detection.
    - The model calculates the probability of the outcome using a logistic function and then classifies the data based on a threshold.
    - Logistic regression is a simple and interpretable model that can handle both numerical and categorical features.
2. Random Forest Classifier
    - Random Forest is a machine learning algorithm that is used for both classification and regression tasks.
    - It is an ensemble learning method that combines multiple decision trees to make predictions.
    - Each decision tree in the random forest is trained on a random subset of the training data and uses a random subset of features for splitting.
    - The final prediction is made by aggregating the predictions of all the individual trees.
    - Random Forest is known for its ability to handle high-dimensional data, handle missing values, and reduce overfitting.
3. Support Vector Machine (SVM)
    - SVM is a type of supervised machine learning algorithm used for classification and regression tasks.
    - Its primary objective is to find the best hyperplane that separates the data into different classes with the maximum margin.
    - The mathematical formula of SVM maximizes the margin between different classes, while also minimizing the classification error within each class.
4. K Nearest Neighbors (KNN)
    - KNN is a supervised machine learning algorithm that is used for both classification and regression tasks.
    - It is a non-parametric algorithm that makes predictions based on the k closest training examples in the feature space.
    - The algorithm works by calculating the distance between the new data point and all the training data points.
    - The k closest training examples are then used to make predictions.
5. Naive Bayes
    - Naive Bayes is a supervised machine learning algorithm that is used for classification tasks.
    - It is a probabilistic classifier that makes predictions using the Bayes’ theorem.
    - The model assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
    - This is a strong assumption that is rarely true in real-world scenarios.
    - However, the algorithm is simple, fast, and performs well on a wide range of datasets.
6. XGBoost
    - XGBoost is a supervised machine learning algorithm that is used for both classification and regression tasks.
    - It is an ensemble learning method that combines multiple weak learners to create a strong learner.
    - The algorithm works by sequentially adding weak learners to an ensemble, each one correcting its predecessor.
    - XGBoost is known for its ability to handle missing data, feature selection, and reducing overfitting.


## Project Structure

- `customer_churn_prediction.ipynb`: This Jupyter Notebook is where the data is explored and models are tested. It imports the classes defined in `classes.py` to train and test the models.

- `classes.py`: This module defines the `MyInvestigator` class and subclasses for each model type. These classes encapsulate the common functionalities for different machine learning algorithms and allow for easy experimentation with different hyperparameters and data preprocessing techniques.

- `churn_app.py`: This is a Streamlit application that uses one of the trained models (the one that performs best) to predict customer churn. The model is loaded from a pickle file created in the Jupyter Notebook.

- `requirements.txt`: This file lists all the Python dependencies required to run this project.

## How to Run This Project

1. Clone the repository and navigate to the project directory.

2. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook to train and test the models:

    ```bash
    jupyter notebook customer_churn_prediction.ipynb
    ```

4. Run the Streamlit application:

    ```bash
    streamlit run churn_app.py
    ```

## About the Machine Learning Models

The `Investigate` classes in `classes.py` are designed to make it easy to experiment with different machine learning models. Each subclass corresponds to a different model type. These classes are used in the Jupyter Notebook to train and test the models on the customer churn data.

The models that performs the best in terms of precsion, recall and a balance between the tow, are saved as pickle files using joblib. These models are then loaded into the Streamlit application in `churn_app.py` to make predictions base on the user's choice.

## Streamlit App
The Streamlit app is a simple web app that allows users to interactively explore the model's predictions on new data.
The User can choose a Priority Score which determines which model to use in the prediction.
 - The Priority Scores are "Recall", "Precision" and "Balance"
 - If the user chooses Balance, it will use a model with the most balance between Precision and Recall
 - If the user chooses Precision, it will use a model that is more focused on precision (less recall)
 - If the user chooses Recall, it will use a model that is more focused on recall (less precision)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Conclusion

This project demonstrates a complete machine learning workflow, from data exploration and model training to deployment of a predictive model in a web application. It showcases the use of Jupyter Notebook for data analysis and model testing, and Streamlit for creating an interactive web application. If you like this book, give the repo a star ✨.
