
# DiaPredict
### Machine Learning for Diabetes Risk Prediction

**DiaPredict** is a machine-learning-based application developed to explore the use of artificial intelligence in identifying potential diabetes risk from selected health-related indicators.

Built with Python and Streamlit, the application combines a trained machine learning model with an interactive interface that allows users to enter health information and receive a model-generated prediction.

> An educational demonstration of how machine learning can contribute to healthcare innovation.

---

## 1. Project Overview

Diabetes is a significant global public health concern. Identifying individuals who may be at risk is an important step towards encouraging awareness, further assessment and timely medical attention.

DiaPredict explores how machine learning can analyse health-related data and identify patterns associated with diabetes.

The project covers the complete development process, from dataset preparation and model training to model evaluation and application deployment.

## 2. Project Objectives

- Explore the application of machine learning in healthcare.
- Prepare and process a diabetes-related dataset.
- Develop and evaluate predictive machine learning models.
- Identify a suitable model based on its performance.
- Deploy the selected model through an interactive web application.
- Demonstrate the practical application of Python in health-related data science.

## 3. Dataset

The project uses the Pima Indians Diabetes dataset, containing health-related variables used to investigate diabetes outcomes.

The dataset includes:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skinfold thickness |
| Insulin | Serum insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the individual |
| Outcome | Diabetes outcome (0 or 1) |

The dataset was examined and prepared before model development, with particular attention to missing or zero-coded values in selected health measurements.

## 4. Model Development

Python and its machine learning libraries were used to develop the prediction system.

The development process included:

1. Data exploration and understanding.
2. Data cleaning and preprocessing.
3. Handling missing values.
4. Training machine learning models.
5. Comparing model performance.
6. Selecting and saving the trained model.
7. Integrating the model into the Streamlit application.

The project explored Logistic Regression and Random Forest approaches.

## 5. Model Evaluation

Model performance was assessed using several evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Particular attention was given to recall because identifying individuals who may have diabetes is an important consideration in a risk-prediction application.

The model's output should be interpreted as an estimate based on learned patterns, not a clinical diagnosis.

## 6. Application Features

DiaPredict provides an interactive interface through which users can:

- Enter selected health-related information.
- Submit information for model prediction.
- View the resulting prediction.
- Interact with a simplified machine learning application without writing code.

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing |
| Streamlit | Application interface |
| Joblib | Model persistence |
| Jupyter Notebook | Model development and experimentation |

## 8. Repository Structure

```text
DIAPREDICT/
│
├── app.py
├── diabetes_model.pkl
├── Diabetes.csv
├── requirements.txt
└── README.md
```

## 9. Installation and Setup

To run DiaPredict locally, follow these steps.

### Step 1: Clone the repository

```bash
git clone https://github.com/VandefanJohnAdekwagh/DIAPREDICT.git
```

### Step 2: Enter the project directory

```bash
cd DIAPREDICT
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
streamlit run app.py
```

The application should open in your default browser.

## 10. Limitations

DiaPredict is an educational machine learning project and is not intended to replace medical diagnosis or professional healthcare services.

Its predictions are limited by the dataset used for development, the variables available and the performance of the trained model.

The application should not be used as the sole basis for medical decisions.

## 11. Future Improvements

Potential improvements include:

- Training with larger and more diverse datasets.
- Exploring additional machine learning algorithms.
- Improving model generalisation.
- Enhancing the user interface.
- Conducting further validation using independent datasets.
- Exploring integration with broader healthcare data systems.

## 12. Project Author

**Vandefan John Adekwagh**

Developed as part of an Artificial Intelligence and Machine Learning training project at Torbita Computer Training School.

---

**Project Repository:**  
https://github.com/VandefanJohnAdekwagh/DIAPREDICT

**Project Name:** DiaPredict  
**Category:** Artificial Intelligence / Machine Learning / Healthcare
