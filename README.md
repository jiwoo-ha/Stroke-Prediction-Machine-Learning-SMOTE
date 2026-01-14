# Stroke Prediction Using Machine Learning and SMOTE-NC

This project explores the effectiveness of various machine learning models in predicting stroke risk using clinical data. A major challenge in this dataset is class imbalance, which is addressed using SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features).

## 🔍 Dataset

- Source: [Cerebral Stroke Dataset on Kaggle](https://www.kaggle.com/datasets/viviansam/cerebral-stroke-dataset)
- Sample size: ~43,400 records with 11 predictor variables and one categorical response variable.
- Features include demographic and clinical variables such as age, hypertension, heart disease, and glucose levels.

## 🧠 Models Used

1. Logistic Regression  
2. Random Forest  
3. CatBoost 
4. XGBoost
5. LightGBM

## ⚙️ Methodology

- **Preprocessing**: Categorical encoding, missing value imputation (BMI and smoking status), feature scaling
- **Class Imbalance Handling**: Applied SMOTE-NC to balance the minority class (stroke cases)
- **Evaluation Metrics**: ROC-AUC, F1-score, confusion matrix, and Sensitivity
- **Model Interpretation**: Feature importance analysis using built-in methods (e.g., SHAP for Tree-based models)

## 📈 Results Summary

- CatBoost and Logistic Regression showed the best performance in terms of AUC and Sensitivity.
- Age, BMI, and average glucose level were the most important predictors across models.

## 🧪 Future Work

- In the current logistic regression model, some potentially nonlinear features such as average glucose level were treated linearly while BMI was modeled with a nonlinear transformation. Incorporating transformed terms (e.g., polynomial or spline basis functions) for additional variables may improve the model’s ability to capture complex relationships and enhance its competitiveness relative to more flexible machine learning models.

- Alternative Imbalance Handling Techniques: Our analysis relied solely on SMOTE-NC to address class imbalance. Exploring additional approaches—such as bootstrap-based methods (e.g., ROSE) or cost-sensitive learning—may provide a more nuanced understanding of how different oversampling strategies affect model performance.

- Model Calibration: Future iterations may incorporate calibration techniques (e.g., Platt scaling or isotonic regression) to improve the interpretability and reliability of predicted probabilities, particularly in clinical decision-making contexts.

## 📁 Repository Structure
├── dataset/ # Raw and processed datasets

├── Rcode/ # R code for preprocessing and model training

├── results/ # Output plots, metrics, and reports

└── README.md # Project overview
