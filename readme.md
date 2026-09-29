# ❤️ Heart Disease Prediction Using Machine Learning

A machine learning project that analyzes patient health data and predicts the likelihood of heart disease using **Logistic Regression** and **Decision Tree Classification**.

The project includes data preprocessing, exploratory data analysis, feature encoding, missing-value handling, model training, evaluation, visualization, and saving trained models for future predictions.

---

## 📌 Project Overview

Heart disease prediction is a binary classification problem where patient health-related information is used to predict whether heart disease is present.

This project uses a heart disease dataset containing patient attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, exercise-induced angina, and other clinical features.

The complete workflow implemented in the project is:

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Duplicate Removal
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Imputation & Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
New Patient Prediction
```

---

## 🎯 Objectives

* Analyze heart disease patient data.
* Clean and preprocess the dataset.
* Handle missing values.
* Convert categorical features into numerical values.
* Explore feature relationships using visualizations.
* Train classification models.
* Compare model performance using evaluation metrics.
* Save trained models and preprocessing objects.
* Use the trained models to make predictions for new patient data.

---

## 📊 Dataset

The project uses the `heart.csv` dataset.

The original dataset contains:

* **920 records**
* **16 columns**

The dataset includes the following attributes:

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Patient age                           |
| `sex`      | Patient sex                           |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Cholesterol level                     |
| `fbs`      | Fasting blood sugar                   |
| `restecg`  | Resting electrocardiographic result   |
| `thalch`   | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression induced by exercise     |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia                           |
| `target`   | Heart disease class                   |

The `id` and `dataset` columns are removed during preprocessing because they are not used as predictive features.

---

## 🧹 Data Preprocessing

The notebook performs several preprocessing steps before training the models.

### 1. Remove unnecessary columns

The `id` and `dataset` columns are removed:

```python
df = df.drop(columns=[c for c in ["id", "dataset"] if c in df.columns])
```

### 2. Handle invalid/missing values

Values such as:

```text
?
" "
""
```

are converted to `NaN`.

### 3. Identify the target column

The notebook searches for possible target column names including:

```text
target
num
HeartDisease
heartdisease
output
condition
class
diagnosis
```

The dataset uses `num` as the original target column.

### 4. Handle numerical missing values

Numerical columns are converted to numeric values and missing values are filled using the column median.

### 5. Handle categorical missing values

Missing categorical values are replaced using the mode of the corresponding column.

### 6. Encode categorical features

Categorical columns are converted into numerical values using `LabelEncoder`.

Categorical features include:

```text
sex
cp
fbs
restecg
exang
slope
thal
```

### 7. Convert target into binary classification

If the target contains more than two classes, values greater than zero are converted to:

```text
0 → No Heart Disease
1 → Heart Disease
```

### 8. Remove duplicate records

The preprocessing pipeline removes duplicate rows.

The notebook reports that **2 duplicate rows** were removed.

After preprocessing, the target distribution is:

```text
No Heart Disease: 410
Heart Disease:    508
```

---

## 📈 Exploratory Data Analysis

The project generates several visualizations to understand the dataset and model performance.

### Class Balance

The project generates:

```text
class_balance.png
```

This visualization shows the distribution between:

* No Heart Disease
* Heart Disease

### Correlation Heatmap

The project generates:

```text
correlation_heatmap.png
```

The heatmap visualizes correlations between the numerical features.

### Confusion Matrix

The project generates:

```text
confusion_matrices.png
```

Confusion matrices are created for both trained models.

### ROC Curves

The project generates:

```text
roc_curves.png
```

ROC curves are used to compare the classification performance of the models.

---

## 🤖 Machine Learning Models

Two supervised machine learning classification algorithms are implemented.

### 1. Logistic Regression

Logistic Regression is used as a classification model for predicting whether heart disease is present.

Configuration used:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### 2. Decision Tree Classifier

A Decision Tree is also trained for classification.

Configuration used:

```python
DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
```

---

## 🔄 Train-Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This creates:

* **80% training data**
* **20% testing data**

The split uses `stratify=y` so that the class distribution is maintained between the training and testing sets.

---

## ⚙️ Feature Processing

Before model training, the project performs:

### Missing Value Imputation

```python
SimpleImputer(strategy="median")
```

### Feature Scaling

```python
StandardScaler()
```

The scaler is fitted on the training data and then applied to the test data.

This helps ensure that features with different numerical ranges are placed on a comparable scale.

---

## 📊 Model Performance

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* ROC-AUC

### Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   79.35% |    80.19% | 83.33% |   81.73% |
| Decision Tree       |   82.07% |    86.32% | 80.39% |   83.25% |

The values above are generated directly from the project's notebook evaluation results.

---

## 💾 Saved Models and Artifacts

The project saves the trained models and preprocessing objects using `joblib`.

### Models

```text
logistic_regression_model.pkl
decision_tree_model.pkl
```

### Preprocessing Objects

```text
scaler.pkl
imputer.pkl
label_encoders.pkl
feature_order.pkl
```

These files allow the same preprocessing steps to be reused when making predictions on new patient data.

---

## 🔮 Making Predictions

The project defines a `predict_patient()` function that accepts:

* trained model
* scaler
* imputer
* label encoders
* patient information
* feature order

The function performs:

```text
Patient Data
     ↓
Categorical Encoding
     ↓
Missing Value Imputation
     ↓
Feature Scaling
     ↓
Model Prediction
     ↓
Prediction + Probability
```

The function returns:

```python
pred, proba
```

where:

* `pred` is the predicted class.
* `proba` is the predicted probability of the positive class.

---

## 🗂️ Project Structure

```text
Heart-Disease/
│
├── .ipynb_checkpoints/
│
├── .env
│
├── Heart_Disease_Prediction_Report.docx
│
├── heart.csv
│
├── heart_disease.ipynb
│
├── class_balance.png
│
├── correlation_heatmap.png
│
├── confusion_matrices.png
│
├── roc_curves.png
│
├── model_results_summary.csv
│
├── logistic_regression_model.pkl
│
├── decision_tree_model.pkl
│
├── scaler.pkl
│
├── imputer.pkl
│
├── label_encoders.pkl
│
└── feature_order.pkl
```

This structure reflects the files currently present in the repository.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/vaishnaviD1509/Heart-Disease.git
```

### 2. Navigate to the project directory

```bash
cd Heart-Disease
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

---

## ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
heart_disease.ipynb
```

Run the notebook cells sequentially.

The notebook will:

1. Load the dataset.
2. Clean the data.
3. Handle missing values.
4. Encode categorical features.
5. Remove duplicates.
6. Perform exploratory analysis.
7. Split the dataset.
8. Preprocess the features.
9. Train Logistic Regression.
10. Train Decision Tree.
11. Evaluate both models.
12. Generate visualizations.
13. Save trained models.
14. Perform a sample patient prediction.

---

## 📌 Sample Prediction

The notebook includes a sample patient prediction.

For the example used in the notebook, the models produced:

```text
Logistic Regression
Result: No Heart Disease
Probability of disease: 47.05%

Decision Tree
Result: No Heart Disease
Probability of disease: 38.81%
```

These values are the output for the specific example patient used in the notebook, not a general medical conclusion.

---

## 📊 Evaluation Metrics

### Accuracy

Measures the percentage of predictions that are correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1 Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

---

## 🔐 Security

The repository currently contains a `.env` file. Do **not** store real API keys, passwords, tokens, or other secrets in a public repository.

Add this to `.gitignore`:

```gitignore
.env
.ipynb_checkpoints/
__pycache__/
*.pyc
```

If a real secret has already been committed, remove it from the repository history and rotate/revoke the exposed credential.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

The predictions produced by the machine learning models should **not be treated as a medical diagnosis or medical advice**. Real-world medical prediction systems require clinical validation, appropriate datasets, expert review, and regulatory considerations.

---

## 🚀 Future Improvements

Possible improvements include:

* Hyperparameter tuning.
* Testing additional classification algorithms.
* Cross-validation.
* Better handling of categorical variables using techniques such as One-Hot Encoding.
* Feature selection and engineering.
* Model explainability using SHAP or similar techniques.
* Building a user-friendly web interface.
* Deploying the prediction system as a web application.
* Testing the model on an independent external dataset.
* Improving the preprocessing pipeline using a Scikit-learn `Pipeline`.

---

## 📚 Project Outputs

The project produces:

```text
class_balance.png
correlation_heatmap.png
confusion_matrices.png
roc_curves.png
model_results_summary.csv
```

These outputs provide visual and numerical information about the dataset and trained models.

---

## 👩‍💻 Author

**Vaishnavi D**

GitHub:
https://github.com/vaishnaviD1509

---

## ⭐ Project Summary

**Heart Disease Prediction** is an end-to-end machine learning project that demonstrates how structured patient data can be cleaned, analyzed, transformed, and used to train classification models.

The project covers the complete machine learning workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Encoding
      ↓
Missing Value Handling
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Persistence
      ↓
New Patient Prediction
```

The project demonstrates practical use of **Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Joblib** in a healthcare-oriented machine learning application.
