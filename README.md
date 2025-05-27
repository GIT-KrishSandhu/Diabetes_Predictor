# 🧠 Diabetes Prediction Using Logistic Regression

This project implements a machine learning classification model to predict the likelihood of diabetes based on medical diagnostic data. It uses the Pima Indians Diabetes Dataset and applies **Logistic Regression** for prediction.

---

## 📁 Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, typically found on platforms like Kaggle or UCI ML Repository. It includes the following features:

- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age in years
- `Outcome`: Class variable (0 = No diabetes, 1 = Diabetes)

---

## ⚙️ How It Works

1. **Data Preprocessing**: The dataset is loaded and split into features and labels.
2. **Model Training**: A logistic regression model is trained on 80% of the data.
3. **Evaluation**: The model is evaluated using accuracy, confusion matrix, classification report, and ROC-AUC curve.
4. **Prediction**: The user is prompted to input personal medical data, and the model predicts diabetes likelihood.

---

## 📊 Visualizations

- **Confusion Matrix**: Understands false positives/negatives and overall classification quality.
- **ROC Curve**: Visualizes the trade-off between sensitivity and specificity.

---

## 🧪 Example Output

```bash
Enter your health details to predict diabetes:
Enter Pregnancies: 2
Enter Glucose: 120
Enter BloodPressure: 70
Enter SkinThickness: 20
Enter Insulin: 80
Enter BMI: 28.5
Enter DiabetesPedigreeFunction: 0.5
Enter Age: 35

✅ The model predicts that you *do not have diabetes*.
```
---

## 📈 Model Performance
 - Accuracy: ~78–82% (may vary depending on random state and data split)
 - ROC-AUC: Shows good separation between classes

---

## 🧰 Requirements
 - Python 3.13
 - pandas
 - numpy
 - scikit-learn
 - flask
 - matplotlib
 - seaborn

---

You can install dependencies with:
``` bash
pip install -r requirements.txt
```
---*-

## 👨‍💻 Authors
 - jahnavi1523 (202401100300130)
 - KAVYATRIPATHI-GIF (202401100300135)
 - Krish6004 (202401100300137)
 - GIT-KrishSandhu (202401100300138)
