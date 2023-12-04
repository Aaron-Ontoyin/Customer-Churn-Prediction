import streamlit as st
import joblib
import os
import pandas as pd

# Importing the MyXGBInvestigator class from the 'classes' module, even though it is not explicitly used in this code.
# The reason is that joblib reads the MyXGBInvestigator class from a file, and therefore the class blueprint needs to be present.
from classes import MyXGBInvestigator


encode_dict: dict = {
    "Female": 0,
    "Male": 1,
    "No": 0,
    "Yes": 1,
    "No phone service": 0,
    "No internet service": 0,
}
bin_cols: list = [
    "Gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "SeniorCitizen",
]
non_bin_categorical_cols: list = [
    "InternetService",
    "Contract",
    "PaymentMethod",
]
continuos_cols: list = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def encode_cat_column(data: pd.DataFrame, column: str, categories: list) -> None:
    """
    Encode categorical column with one-hot encoding.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to encode.
        categories (list): The list of categories for the column.
    """
    for category in categories:
        col_name = f"{column}_{category}"
        data[col_name] = [1 if data[column][0] == category else 0]
    data.drop(column, axis=1, inplace=True)


def app() -> None:
    """
    Streamlit application for predicting customer churn.
    """
    st.title("Predict Customer Churn")
    st.write(
        "This is a simple app to predict customer churn. "
        "The data was gotten from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)."
    )
    # Choose precission, recall or balance
    st.sidebar.subheader("Choose Priority Score")
    priority_score = st.sidebar.selectbox(
        "Choose Priority Score", ["Precision", "Recall", "Balance"]
    )
    # Describe what the chosen priority score means
    if priority_score == "Precision":
        st.sidebar.write(
            "Precision is a measure of the accuracy of a classification model. "
            "It quantifies the proportion of correctly predicted positive instances out of the total instances predicted as positive. "
            "In simpler terms, precision tells us how well the model performs in correctly identifying positive cases. "
            "A high precision value indicates a low false positive rate, meaning that the model is good at avoiding false alarms."
            "\n This choice makes the model have a high precission, this is at the expense of recall!"
        )
    elif priority_score == "Recall":
        st.sidebar.write(
            "Recall (sensitivity) is a measure of the completeness of a classification model. "
            "It quantifies the proportion of correctly predicted positive instances out of all the actual positive instances. "
            "In simpler terms, recall tells us how well the model is able to identify all the positive cases. "
            "A high recall value indicates a low false negative rate, meaning that the model is good at avoiding false negatives."
            "\n This choice makes the model have a high recall, this is at the expense of precission!"
        )
    elif priority_score == "Balance":
        st.sidebar.write(
            "This choice makes the model have a balance between precission and recall!"
        )

    # The base directory for this project is not '.' but the location where the virtual environment is stored, as it is shared with other projects.
    model_dict_path: str = os.path.join(
        os.path.dirname(__file__), f"model_dict_{priority_score.lower()}.pkl"
    )
    model_dict: dict = joblib.load(model_dict_path)

    dropped_columns = model_dict["dropped_columns"]
    if dropped_columns:
        st.info(f"Do not bother about {dropped_columns}. They would be dropped!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input(
            "TotalCharges",
            min_value=0.0,
            max_value=10000.0,
            step=0.01,
            key="totalcharges",
        )
        st.number_input(
            "MonthlyCharges",
            min_value=0.0,
            max_value=10000.0,
            step=0.01,
            key="monthlycharges",
        )
        st.number_input(
            "Tenure", min_value=0.0, max_value=10000.0, step=0.01, key="tenure"
        )
        st.selectbox(
            "OnlineSecurity", ["No", "Yes", "No internet service"], key="onlinesecurity"
        )
        st.selectbox("PaperlessBilling", ["No", "Yes"], key="paperlessbilling")
        st.selectbox("SeniorCitizen", ["No", "Yes"], key="seniorcitizen")
    with col2:
        st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"], key="contract"
        )
        st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="paymentmethod",
        )
        st.selectbox(
            "InternetService", ["DSL", "Fiber optic", "No"], key="internetservice"
        )
        st.selectbox(
            "OnlineBackup", ["No", "Yes", "No internet service"], key="onlinebackup"
        )
        st.selectbox(
            "MultipleLines", ["No", "Yes", "No phone service"], key="multiplelines"
        )
        st.selectbox("PhoneService", ["No", "Yes"], key="phoneservice")
    with col3:
        st.selectbox(
            "DeviceProtection",
            ["No", "Yes", "No internet service"],
            key="deviceprotection",
        )
        st.selectbox(
            "TechSupport", ["No", "Yes", "No internet service"], key="techsupport"
        )
        st.selectbox(
            "StreamingTV", ["No", "Yes", "No internet service"], key="streamingtv"
        )
        st.selectbox(
            "StreamingMovies",
            ["No", "Yes", "No internet service"],
            key="streamingmovies",
        )
        st.selectbox("Partner", ["No", "Yes"], key="partner")
        st.selectbox("Dependents", ["No", "Yes"], key="dependents")
        st.selectbox("Gender", ["Male", "Female"], key="gender")

    if st.button("Predict"):
        with st.spinner("Making Prediction..."):
            data = pd.DataFrame(
                data={
                    "TotalCharges": [st.session_state.totalcharges],
                    "MonthlyCharges": [st.session_state.monthlycharges],
                    "Tenure": [st.session_state.tenure],
                    "OnlineSecurity": [st.session_state.onlinesecurity],
                    "Contract": [st.session_state.contract],
                    "PaymentMethod": [st.session_state.paymentmethod],
                    "InternetService": [st.session_state.internetservice],
                    "OnlineBackup": [st.session_state.onlinebackup],
                    "DeviceProtection": [st.session_state.deviceprotection],
                    "TechSupport": [st.session_state.techsupport],
                    "StreamingTV": [st.session_state.streamingtv],
                    "StreamingMovies": [st.session_state.streamingmovies],
                    "MultipleLines": [st.session_state.multiplelines],
                    "PaperlessBilling": [st.session_state.paperlessbilling],
                    "SeniorCitizen": [st.session_state.seniorcitizen],
                    "PhoneService": [st.session_state.phoneservice],
                    "Partner": [st.session_state.partner],
                    "Dependents": [st.session_state.dependents],
                    "Gender": [st.session_state.gender],
                }
            )
            data[bin_cols] = data[bin_cols].applymap(
                lambda x: x if x in [0, 1] else encode_dict.get(x)
            )

            encode_cat_column(data, "Contract", ["One year", "Two year"])
            encode_cat_column(
                data,
                "PaymentMethod",
                ["Credit card (automatic)", "Electronic check", "Mailed check"],
            )
            encode_cat_column(data, "InternetService", ["Fiber optic", "No"])

            st.subheader("You entered this Data")
            data

            pipeline = model_dict["model"]
            prediction = pipeline.predict(data)[0]

            if prediction == 0:
                st.success("Customer will not churn")
            else:
                st.error("Customer will churn")


if __name__ == "__main__":
    app()
