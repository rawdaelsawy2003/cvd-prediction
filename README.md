Web application link: http://127.0.0.1:5000

Explaining the project & web tutorial: https://drive.google.com/drive/folders/1a-gwxm7zf1-V6A0Rp_PSYicIGFwgCLK7

Rawda Elsawy, Machine learning course

Project Overview:
Cardiovascular disease is the number one cause of death worldwide, responsible for nearly 18 million deaths every year. In this project, I built a complete end-to-end machine learning pipeline that predicts whether a patient has cardiovascular disease
based on their clinical measurements and lifestyle habits.

Research Questions:
1-Can we predict whether a patient has cardiovascular disease based on clinical and lifestyle features?
2- Which features (age, cholesterol, blood pressure, BMI) are the strongest predictors of CVD?
3- How do blood pressure and cholesterol interact with age to affect disease risk?

Dataset
Name: Cardiovascular Disease Dataset
Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
Author: Svetlana Ulianova
Format: CSV (semicolon-separated)
Rows= 70,000
Columns= 12

The dataset contains the following columns:
•id : Patient ID
•age : Age in years 
•gender : 1 = Female, 2 = Male
•height : Height in cm
•weight : Weight in kg
•ap_hi : Systolic blood pressure
•ap_lo : Diastolic blood pressure
•cholesterol : 1 = Normal, 2 = Above Normal, 3 = Well Above Normal
•gluc : Glucose level: 1 = Normal, 2 = Above Normal, 3 = Well Above Normal
•smoke : Smoker: 0 or 1
•alco : Alcohol intake: 0 or 1
•active : Physical activity: 0 or 1
•cardio : Target variable: 1 = CVD present, 0 = CVD absent

Why This Dataset is NOT Clean:
1-Blood pressure errors : ap_hi contains values like -150 and 1000+ (clinically impossible)
2-Height outliers : values as low as 55 cm present
3-Weight outliers : values of 10 kg and 200+ kg present
4-Age in days : must be converted to years
5-No engineered features : BMI, pulse pressure, hypertension flags need to be derived
6-Inconsistencies : data collected from three different examination procedures

Project Workflow:
Step 1 : Data Collection: Downloaded the raw dataset from Kaggle.
Step 2 : Data Cleaning: Removed duplicate rows and filtered out impossible values in blood pressure, height, and weight columns.
Step 3 : Feature Engineering: Created 5 new meaningful features from existing data.
Step 4 : Exploratory Data Analysis: Used 7 different plot types to investigate patterns and answer research questions with both univariate and bivariate analysis.
Step 5 : Feature Selection: Applied SelectKBest Filter Method using ANOVA F-test to select the top 10 most relevant features.
Step 6 : Model Training: Trained and compared 3 different machine learning algorithms.
Step 7 : Hyperparameter Tuning: Used GridSearchCV with 5-fold cross-validation to find the best model configuration.
Step 8 : Evaluation: Evaluated the final model using Precision, Recall, F1-Score, Accuracy, and ROC-AUC.
Step 9 : Deployment: Built and deployed a Flask web application for live predictions.

Feature Engineering:
I created 5 new features to enrich the dataset:
-bmi : Body Mass Index calculated as weight divided by height squared.
-age_years : Age converted from days to years.
-pulse_pressure : Difference between systolic and diastolic blood pressure.
-hypertension : Set to 1 if blood pressure is at or above 140/90.
-obese : Set to 1 if BMI is 30 or above.

Models Compared:
I trained and compared three algorithms. Logistic Regression was used as the baseline and achieved an accuracy of around 0.72. Random Forest achieved an accuracy of around 0.73 with strong precision and recall. Gradient Boosting achieved similar results to Random Forest. The Tuned Random Forest was selected as the final model after GridSearchCV optimization, achieving the best overall performance with accuracy around 0.74, precision around 0.73, recall around 0.73, F1 around 0.73, and ROC-AUC around 0.81.

The project folder contains the following files:

-cardiovascular_disease_project.ipynb : Full notebook with all steps
-app.py : Flask web application
-model.pkl : Saved trained model
-scaler.pkl : Saved StandardScaler transformer
-features.pkl : Saved list of selected features
-dataset_before_cleaning.csv : Original raw dataset
-dataset_after_cleaning.csv : Cleaned and engineered dataset
-requirements.txt : Python dependencies
-README.md : This file

How to Run:
Step 1 : Install all required libraries by running: pip install -r requirements.txt
Step 2 : Open the notebook cardiovascular_disease_project.ipynb and run all cells from top to bottom.
Step 3 : Launch the web app by running: python app.py then open http://127.0.0.1:5000 in your browser.

Web App:
The web application allows any user to enter their health measurements and receive an instant cardiovascular disease risk prediction. The app includes a patient input form covering age, blood pressure, cholesterol, BMI, and lifestyle habits. It returns either a High Risk or Low Risk result along with an animated probability bar. The app is built using Flask and powered by the trained Random Forest model.