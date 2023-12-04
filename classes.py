"""
This module contains the definition of the 'Investigator' classes and related methods for analyzing customer churn prediction.

Classes:
- `MyInvestigator`: Represents an investigator for analyzing customer churn prediction.
Provides methods for training and evaluating machine learning models on a given dataset. This is the Base class.
- `MyLogisticInvestigator`: Represents an investigator for analyzing customer churn prediction using Logistic Regression.
- `MySVMInvestigator`: Represents an investigator for analyzing customer churn prediction using Support Vector Machine (SVM).
- `MyForestInvestigator`: Represents an investigator for analyzing customer churn prediction using Random Forest.
- `MyBayesInvestigator`: Represents an investigator for analyzing customer churn prediction using Naive Bayes.
- `MyKNNInvestigator`: Represents an investigator for analyzing customer churn prediction using K-Nearest Neighbors (KNN).
- `MyXGBInvestigator`: Represents an investigator for analyzing customer churn prediction using XGBoost.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt


class MyInvestigator:
    """
    Represents an investigator for analyzing customer churn prediction.
    Provides methods for training and evaluating machine learning models on a given dataset.

    Methods:
    - __init__: Initialize MyInvestigator object.
    - drop: Drop columns from the input DataFrame.
    - evaluate_model: Evaluate the performance of a model.
    - SMOT: Perform Synthetic Minority Over-sampling Technique (SMOTE) on the data.
    - ROS: Perform Random Over-sampling (ROS) on the data.
    - RUS: Perform Random Under-sampling (RUS) on the data.
    - print_report: Print the report dictionary.
    - _investigate: Perform investigation on the data.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> None:
        """
        Initialize MyInvestigator object.

        MyInvestigator is a class that represents an investigator for analyzing customer churn prediction.
        It provides methods for training and evaluating machine learning models on a given dataset.
        This is the Base Class from which different model types Investigator classes would inherit.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            test_size (float, optional): Proportion of the dataset to include in the test split.
                Defaults to 0.2.
        """
        self.model_class: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.drop_columns: Optional[List[str]] = None
        self.X_train: pd.DataFrame
        self.X_val: pd.DataFrame
        self.y_train: pd.Series
        self.y_val: pd.Series
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=11
        )

    def drop(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns from the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with dropped columns.
        """
        if self.drop_columns:
            return X.drop(self.drop_columns, axis=1)
        return X

    @staticmethod
    def evaluate_model(
        model: Any, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
        """
        Evaluate the performance of a model.

        Args:
            model (Any): Trained model / Pipeline.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.

        Returns:
            Tuple[pd.DataFrame, Optional[plt.Figure]]: Evaluation metrics and ROC curve figure.
        """
        y_pred = model.predict(X_val)
        print(y_pred)

        try:
            y_proba = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_proba)

            fpr, tpr, _ = roc_curve(y_val, y_proba)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend(loc="lower right")
            plt.close(fig)

        except AttributeError:
            y_proba = "Model Proba is False"
            roc_auc = None
            fig = None

        cm = confusion_matrix(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC AUC",
                    "confusion matrix",
                ],
                "Score": [precision, recall, f1, roc_auc, cm],
            }
        )

        return metrics, fig

    @staticmethod
    def SMOTE(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform Synthetic Minority Over-sampling Technique (SMOTE) on the data.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Oversampled features and target variable.
        """
        smote = SMOTE(random_state=2)
        X_sm, y_sm = smote.fit_resample(X, y)
        return X_sm, y_sm

    @staticmethod
    def ROS(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform Random Over-sampling (ROS) on the data.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Oversampled features and target variable.
        """
        ros = RandomOverSampler(random_state=2)
        X_ros, y_ros = ros.fit_resample(X, y)
        return X_ros, y_ros

    @staticmethod
    def RUS(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform Random Under-sampling (RUS) on the data.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Undersampled features and target variable.
        """
        rus = RandomUnderSampler(random_state=2)
        X_rus, y_rus = rus.fit_resample(X, y)
        return X_rus, y_rus

    @staticmethod
    def print_report(report_dic: Dict[str, Any]) -> None:
        """
        Print the report dictionary.

        Args:
            report_dic (Dict[str, Any]): Report dictionary.
        """
        if report_dic["scaler"]:
            print("Scaler: ", report_dic["scaler"].__name__)
        else:
            print("Scaler: ", report_dic["scaler"])

        print("Drop columns: ", report_dic["drop_cols"])

        if report_dic["handle_inbalance"]:
            print("Handle inbalance: ", report_dic["handle_inbalance"].__name__)
        else:
            print("Handle inbalance: ", report_dic["handle_inbalance"])

        if report_dic["totalChargesSkewTransfromer"]:
            print(
                "TotalChargesSkewTransfromer: ",
                report_dic["totalChargesSkewTransfromer"].__name__,
            )
        else:
            print(
                "TotalChargesSkewTransfromer: ",
                report_dic["totalChargesSkewTransfromer"],
            )

        print("Scoring: ", report_dic["scoring"])

    def _investigate(
        self,
        model_: Any,
        scaler: Optional[StandardScaler] = None,
        totalChargesSkewTransfromer: Optional[Any] = None,
        param_grid: Optional[Dict[str, Any]] = None,
        drop_cols: Optional[List[str]] = None,
        handle_inbalance: Optional[Any] = None,
        scoring: str = "recall",
        model_type: str = "normal",
        model__n_estimators: int = 50,
        model__learning_rate: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform investigation on the data.

        Args:
            model_ (Any): Model to be used.
            scaler (Optional[StandardScaler], optional): Scaler for preprocessing. Defaults to None.
            totalChargesSkewTransfromer (Optional[Any], optional): Transformer for TotalCharges feature. Defaults to None.
            param_grid (Optional[Dict[str, Any]], optional): Grid of hyperparameters for grid search. Defaults to None.
            drop_cols (Optional[List[str]], optional): Columns to be dropped. Defaults to None.
            handle_inbalance (Optional[Any], optional): Method for handling class imbalance. Defaults to None.
            scoring (str, optional): Scoring metric for evaluation. Defaults to "recall".
            model_type (str, optional): Type of model. Defaults to "normal".
            model__n_estimators (int, optional): Number of estimators for ensemble models. Defaults to 50.
            model__learning_rate (int, optional): Learning rate for AdaBoost or some other models that requires it. Defaults to 1.

        Returns:
            Dict[str, Any]: Dictionary containing the investigation results.
        """
        self.drop_columns = drop_cols
        self.scaler = scaler() if scaler else None
        if totalChargesSkewTransfromer:
            self.X_val["TotalCharges"] = totalChargesSkewTransfromer(
                self.X_val["TotalCharges"]
            )
            self.X_train["TotalCharges"] = totalChargesSkewTransfromer(
                self.X_train["TotalCharges"]
            )
        if handle_inbalance:
            self.X_train, self.y_train = handle_inbalance(self.X_train, self.y_train)
        if model_type == "bagging":
            model_ = BaggingClassifier(
                estimator=model_, n_estimators=model__n_estimators, random_state=42
            )
        if model_type == "adaboost":
            model_ = AdaBoostClassifier(
                estimator=model_,
                n_estimators=model__n_estimators,
                random_state=42,
                learning_rate=model__learning_rate,
            )

        transformers = [
            (
                "drop_total_charges",
                FunctionTransformer(self.drop),
                self.X_train.columns,
            )
        ]
        if self.scaler is not None:
            transformers.append(("scale", self.scaler, self.X_train.columns))
        preprocessor = ColumnTransformer(transformers=transformers)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model_),
            ]
        )

        if param_grid:
            grid_search = GridSearchCV(
                pipeline, param_grid=param_grid, cv=5, verbose=2, scoring=scoring
            )
            grid_search.fit(self.X_train, self.y_train)
            return {
                "model_gs": grid_search,
                "scaler": scaler,
                "drop_cols": drop_cols,
                "handle_inbalance": handle_inbalance,
                "totalChargesSkewTransfromer": totalChargesSkewTransfromer,
                "scoring": scoring,
            }

        else:
            model = pipeline.fit(self.X_train, self.y_train)
            return {
                "model": model,
                "scaler": scaler,
                "drop_cols": drop_cols,
                "handle_inbalance": handle_inbalance,
                "totalChargesSkewTransfromer": totalChargesSkewTransfromer,
                "scoring": scoring,
            }


from sklearn.linear_model import LogisticRegression


class MyLogisticInvestigator(MyInvestigator):
    """
    Represents an investigator for analyzing customer churn prediction using Logistic Regression.
    """

    def __init__(self, X, y, test_size=0.2) -> None:
        """
        Initialize MyLogisticInvestigator object.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        """
        super().__init__(X, y, test_size)
        self.model_class = LogisticRegression

    def investigate(
        self,
        scaler,
        totalChargesSkewTransfromer=None,
        param_grid=None,
        scoring="recall",
        drop_cols=None,
        handle_inbalance=None,
        model__C=1,
        model__penalty="l2",
        model__solver="lbfgs",
        model__max_iter=3000,
        model__class_weight=None,
        model_type="normal",
        model__n_estimators=50,
        model__learning_rate=1,
    ):
        """
        Perform investigation using logistic regression model.

        Args:
            scaler: The scaler object used to preprocess the data.
            totalChargesSkewTransfromer: The transformer object used to handle skewness in the 'totalCharges' feature.
            param_grid: The parameter grid for hyperparameter tuning.
            scoring (str, optional): The scoring metric used for model evaluation. Defaults to "recall".
            drop_cols: The columns to be dropped from the dataset.
            handle_inbalance: The method used to handle class imbalance.
            model__C (float, optional): Inverse of regularization strength. Smaller values specify stronger regularization. Defaults to 1.
            model__penalty (str, optional): The norm used in the penalization. Defaults to "l2".
            model__solver (str, optional): The algorithm to use in the optimization problem. Defaults to "lbfgs".
            model__max_iter (int, optional): Maximum number of iterations taken for the solvers to converge. Defaults to 3000.
            model__class_weight: Weights associated with classes in the form {class_label: weight}. Defaults to None.
            model_type (str, optional): The type of logistic regression model to use. Defaults to "normal".
            model__n_estimators (int, optional): The number of trees in the forest. Defaults to 50.
            model__learning_rate (float, optional): Learning rate shrinks the contribution of each tree. Defaults to 1.

        Returns:
            dict: Dictionary containing investigation results.
        """
        model_ = LogisticRegression(
            max_iter=model__max_iter,
            C=model__C,
            penalty=model__penalty,
            solver=model__solver,
            class_weight=model__class_weight,
        )
        return self._investigate(
            model_,
            scaler=scaler,
            totalChargesSkewTransfromer=totalChargesSkewTransfromer,
            param_grid=param_grid,
            drop_cols=drop_cols,
            handle_inbalance=handle_inbalance,
            scoring=scoring,
            model_type=model_type,
            model__n_estimators=model__n_estimators,
            model__learning_rate=model__learning_rate,
        )


from sklearn import svm


class MySVMInvestigator(MyInvestigator):
    """
    Represents an investigator for analyzing customer churn prediction using Support Vector Machine (SVM).
    """

    def __init__(self, X, y, test_size=0.2) -> None:
        """
        Initialize MySVMInvestigator object.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        """
        super().__init__(X, y, test_size)
        self.model_class = svm.SVC

    def investigate(
        self,
        scaler,
        totalChargesSkewTransfromer=None,
        param_grid=None,
        drop_cols=None,
        handle_inbalance=None,
        scoring="recall",
        model__C=1,
        model__kernel="rbf",
        model__max_iter=3000,
        model__class_weight=None,
        model__gamma="scale",
        model__tol=0.001,
        model__degree=3,
        model__learning_rate=1,
        model__n_estimators=50,
        model__probability=False,
        model_type="normal",
    ) -> dict:
        """
        Perform investigation using Support Vector Machine (SVM) model.

        Args:
            scaler: The scaler object used to preprocess the data.
            totalChargesSkewTransfromer: The transformer object used to handle skewness in the 'totalCharges' feature.
            param_grid: The parameter grid for hyperparameter tuning.
            drop_cols: The columns to drop from the dataset.
            handle_inbalance: The method to handle class imbalance.
            scoring (str, optional): The scoring metric for model evaluation. Defaults to "recall".
            model__C (float, optional): The penalty parameter C of the error term. Defaults to 1.
            model__kernel (str, optional): Specifies the kernel type to be used in the algorithm. Defaults to "rbf".
            model__max_iter (int, optional): The maximum number of iterations. Defaults to 3000.
            model__class_weight (dict or 'balanced', optional): The class weights. Defaults to None.
            model__gamma (str or float, optional): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Defaults to "scale".
            model__tol (float, optional): Tolerance for stopping criterion. Defaults to 0.001.
            model__degree (int, optional): Degree of the polynomial kernel function. Defaults to 3.
            model__learning_rate (float, optional): Learning rate for the AdaBoost classifier. Defaults to 1.
            model__n_estimators (int, optional): The number of base estimators in the ensemble. Defaults to 50.
            model__probability (bool, optional): Whether to enable probability estimates. Defaults to False.
            model_type (str, optional): The type of SVM model to use. Defaults to "normal".

        Returns:
            dict: The investigation results.
        """
        model_ = svm.SVC(
            max_iter=model__max_iter,
            C=model__C,
            kernel=model__kernel,
            gamma=model__gamma,
            degree=model__degree,
            probability=model__probability,
            class_weight=model__class_weight,
            tol=model__tol,
        )

        return self._investigate(
            model_,
            scaler=scaler,
            totalChargesSkewTransfromer=totalChargesSkewTransfromer,
            param_grid=param_grid,
            drop_cols=drop_cols,
            handle_inbalance=handle_inbalance,
            scoring=scoring,
            model_type=model_type,
            model__n_estimators=model__n_estimators,
            model__learning_rate=model__learning_rate,
        )


from sklearn.ensemble import RandomForestClassifier


class MyForestInvestigator(MyInvestigator):
    """
    Represents an investigator for analyzing customer churn prediction using Random Forest.
    """

    def __init__(self, X, y, test_size=0.2) -> None:
        """
        Initialize the MyForestInvestigator class.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        """
        super().__init__(X, y, test_size=test_size)
        self.model_class = RandomForestClassifier

    def investigate(
        self,
        scaler,
        totalChargesSkewTransfromer=None,
        param_grid=None,
        drop_cols=None,
        handle_inbalance=None,
        scoring="recall",
        model__criterion="gini",
        model__class_weight=None,
        model__learning_rate=1,
        model__n_estimators=50,
        model_type="normal",
        model__max_samples=None,
        model__max_features=None,
        model__max_depth=None,
        model__max_leaf_nodes=None,
        model__min_samples_leaf=1,
        model__min_samples_split=2,
    ) -> Dict[str, Any]:
        """
        Perform investigation using RandomForestClassifier.

        Args:
            scaler: The scaler object used for feature scaling.
            totalChargesSkewTransfromer: The transformer object used for skew correction of 'TotalCharges' column.
            param_grid: The parameter grid for hyperparameter tuning.
            drop_cols: The columns to drop from the dataset.
            handle_inbalance: The method to handle class imbalance.
            scoring (str, optional): The scoring metric for model evaluation. Defaults to "recall".
            model__criterion (str, optional): The function to measure the quality of a split. Defaults to "gini".
            model__class_weight (str or dict, optional): The class weight for imbalanced datasets. Defaults to None.
            model__learning_rate (float, optional): The learning rate of the model. Defaults to 1.
            model__n_estimators (int, optional): The number of trees in the forest. Defaults to 50.
            model_type (str, optional): The type of model to use. Defaults to "normal".
            model__max_samples (int or float, optional): The maximum number of samples to draw from X to train each base estimator. Defaults to None.
            model__max_features (int, float or str, optional): The number of features to consider when looking for the best split. Defaults to None.
            model__max_depth (int or None, optional): The maximum depth of the tree. Defaults to None.
            model__max_leaf_nodes (int or None, optional): The maximum number of leaf nodes. Defaults to None.
            model__min_samples_leaf (int, float or None, optional): The minimum number of samples required to be at a leaf node. Defaults to 1.
            model__min_samples_split (int, float or None, optional): The minimum number of samples required to split an internal node. Defaults to 2.

        Returns:
            dict: The investigation results.
        """
        model_ = RandomForestClassifier(
            criterion=model__criterion,
            n_estimators=model__n_estimators,
            class_weight=model__class_weight,
            max_samples=model__max_samples,
            max_features=model__max_features,
            max_depth=model__max_depth,
            max_leaf_nodes=model__max_leaf_nodes,
            min_samples_leaf=model__min_samples_leaf,
            min_samples_split=model__min_samples_split,
        )

        return self._investigate(
            model_,
            scaler=scaler,
            totalChargesSkewTransfromer=totalChargesSkewTransfromer,
            param_grid=param_grid,
            drop_cols=drop_cols,
            handle_inbalance=handle_inbalance,
            scoring=scoring,
            model_type=model_type,
            model__n_estimators=model__n_estimators,
            model__learning_rate=model__learning_rate,
        )


from sklearn.naive_bayes import GaussianNB, MultinomialNB


class MyBayesInvestigator(MyInvestigator):
    """
    Represents an investigator for analyzing customer churn prediction using Naive Bayes.
    """

    def __init__(self, X, y, test_size=0.2, type: str = "gaussian") -> None:
        """
        Initialize MyBayesInvestigator object.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            type (str, optional): The type of Naive Bayes model to use. Defaults to "gaussian".
        """
        super().__init__(X, y, test_size=test_size)
        if type == "gaussian":
            self.model_class = GaussianNB
        else:
            self.model_class = MultinomialNB

    def investigate(
        self,
        scaler,
        totalChargesSkewTransfromer=None,
        param_grid=None,
        drop_cols=None,
        handle_inbalance=None,
        scoring: str = "recall",
        model__learning_rate: int = 1,
        model__n_estimators: int = 50,
        model_type: str = "normal",
        model__var_smoothing: float = 1e-9,
        model__priors=None,
        model__force_alpha: bool = True,
    ):
        """
        Perform investigation using Naive Bayes model.

        Args:
            scaler: The scaler object used for feature scaling.
            totalChargesSkewTransfromer: The transformer object used for skewness correction of 'TotalCharges' feature.
            param_grid: The parameter grid for hyperparameter tuning.
            drop_cols: The columns to drop from the dataset.
            handle_inbalance: The method to handle class imbalance.
            scoring (str, optional): The scoring metric for model evaluation. Defaults to "recall".
            model__learning_rate (int, optional): The learning rate for the model. Defaults to 1.
            model__n_estimators (int, optional): The number of estimators for the model. Defaults to 50.
            model_type (str, optional): The type of model to use. Defaults to "normal".
            model__var_smoothing (float, optional): The variance smoothing parameter for GaussianNB. Defaults to 1e-9.
            model__priors: The prior probabilities for MultinomialNB.
            model__force_alpha (bool, optional): Whether to force alpha parameter for MultinomialNB. Defaults to True.

        Returns:
            The investigation results.
        """
        if self.model_class == GaussianNB:
            model_ = GaussianNB(
                var_smoothing=model__var_smoothing, priors=model__priors
            )
        else:
            model_ = MultinomialNB(force_alpha=model__force_alpha)

        return self._investigate(
            model_,
            scaler=scaler,
            totalChargesSkewTransfromer=totalChargesSkewTransfromer,
            param_grid=param_grid,
            drop_cols=drop_cols,
            handle_inbalance=handle_inbalance,
            scoring=scoring,
            model_type=model_type,
            model__n_estimators=model__n_estimators,
            model__learning_rate=model__learning_rate,
        )


from sklearn.neighbors import KNeighborsClassifier


class MyKNNInvestigator(MyInvestigator):
    """
    Represents an investigator for analyzing customer churn prediction using K-Nearest Neighbors (KNN).
    """

    def __init__(self, X, y, test_size=0.2) -> None:
        """
        Initialize the MyKNNInvestigator class.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        """
        super().__init__(X, y, test_size=test_size)
        self.model_class = KNeighborsClassifier

    def investigate(
        self,
        scaler,
        totalChargesSkewTransfromer=None,
        param_grid=None,
        drop_cols=None,
        handle_inbalance=None,
        scoring="recall",
        model__learning_rate=1,
        model__n_estimators=50,
        model_type="normal",
        model__n_neighbors=5,
        model__weights="uniform",
    ):
        """
        Perform investigation using K-Nearest Neighbors algorithm.

        Args:
            scaler: The scaler object used to scale the input features.
            totalChargesSkewTransfromer: The transformer object used to transform the 'totalCharges' column.
            param_grid: The parameter grid for hyperparameter tuning.
            drop_cols: The columns to be dropped from the dataset.
            handle_inbalance: The method to handle class imbalance.
            scoring (str, optional): The scoring metric for model evaluation. Defaults to "recall".
            model__learning_rate (float, optional): The learning rate of the model. Defaults to 1.
            model__n_estimators (int, optional): The number of estimators in the model. Defaults to 50.
            model_type (str, optional): The type of model to be used. Defaults to "normal".
            model__n_neighbors (int, optional): The number of neighbors to consider in the KNN algorithm. Defaults to 5.
            model__weights (str, optional): The weight function used in prediction. Defaults to "uniform".

        Returns:
            dict: The investigation results.
        """
        model_ = KNeighborsClassifier(
            n_neighbors=model__n_neighbors, weights=model__weights
        )

        return self._investigate(
            model_,
            scaler=scaler,
            totalChargesSkewTransfromer=totalChargesSkewTransfromer,
            param_grid=param_grid,
            drop_cols=drop_cols,
            handle_inbalance=handle_inbalance,
            scoring=scoring,
            model_type=model_type,
            model__n_estimators=model__n_estimators,
            model__learning_rate=model__learning_rate,
        )


from xgboost import XGBClassifier


class MyXGBInvestigator(MyInvestigator):
    """
    Represents an investigator for analyzing customer churn prediction using XGBoost.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> None:
        """
        Initialize the MyXGBInvestigator class.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target variable.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        """
        super().__init__(X, y, test_size=test_size)
        self.model_class = XGBClassifier

    def investigate(
        self,
        scaler: StandardScaler,
        totalChargesSkewTransfromer: Optional[Any] = None,
        param_grid: Optional[Dict[str, Any]] = None,
        drop_cols: Optional[List[str]] = None,
        handle_inbalance: Optional[bool] = None,
        scoring: str = "recall",
        model__learning_rate: float = 0.01,
        model__n_estimators: int = 50,
        model__max_depth: int = 3,
        model__min_child_weight: int = 1,
        model__gamma: int = 0,
        model__subsample: float = 1,
        model__colsample_bytree: float = 1,
        model__alpha: int = 0,
        model__lambda: int = 1,
        model_type: str = "normal",
    ) -> Dict[str, Any]:
        """
        Perform investigation using XGBoost classifier.

        Args:
            scaler (StandardScaler): The scaler object for feature scaling.
            totalChargesSkewTransfromer (Optional[Transformer], optional): The transformer object for skewness correction of 'TotalCharges' column. Defaults to None.
            param_grid (Optional[Dict[str, Any]], optional): The parameter grid for hyperparameter tuning. Defaults to None.
            drop_cols (Optional[List[str]], optional): The list of columns to drop from the dataset. Defaults to None.
            handle_inbalance (Optional[bool], optional): Flag to handle class imbalance. Defaults to None.
            scoring (str, optional): The scoring metric for model evaluation. Defaults to "recall".
            model__learning_rate (float, optional): The learning rate of the XGBoost model. Defaults to 0.01.
            model__n_estimators (int, optional): The number of estimators (trees) in the XGBoost model. Defaults to 50.
            model__max_depth (int, optional): The maximum depth of each tree in the XGBoost model. Defaults to 3.
            model__min_child_weight (int, optional): The minimum sum of instance weight (hessian) needed in a child. Defaults to 1.
            model__gamma (int, optional): The minimum loss reduction required to make a further partition on a leaf node of the tree. Defaults to 0.
            model__subsample (float, optional): The subsample ratio of the training instances. Defaults to 1.
            model__colsample_bytree (float, optional): The subsample ratio of columns when constructing each tree. Defaults to 1.
            model__alpha (int, optional): The L1 regularization term on weights. Defaults to 0.
            model__lambda (int, optional): The L2 regularization term on weights. Defaults to 1.
            model_type (str, optional): The type of XGBoost model to use. Defaults to "normal".

        Returns:
            Dict[str, Any]: The investigation results.
        """
        model_ = XGBClassifier(
            n_estimators=model__n_estimators,
            learning_rate=model__learning_rate,
            max_depth=model__max_depth,
            reg_lambda=model__lambda,
            min_child_weight=model__min_child_weight,
            gamma=model__gamma,
            subsample=model__subsample,
            colsample_bytree=model__colsample_bytree,
            alpha=model__alpha,
        )

        return self._investigate(
            model_,
            scaler=scaler,
            totalChargesSkewTransfromer=totalChargesSkewTransfromer,
            param_grid=param_grid,
            drop_cols=drop_cols,
            handle_inbalance=handle_inbalance,
            scoring=scoring,
            model_type=model_type,
            model__n_estimators=model__n_estimators,
            model__learning_rate=model__learning_rate,
        )
