## Rawda Elsawy, Machine learning course

# Cardiovascular Disease Prediction — Final Data Science Project

## Dataset Proposal

**Dataset:** Cardiovascular Disease Dataset  
**Source:** <https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>  
**Author:** Svetlana Ulianova  
**Format:** CSV (semicolon-separated)

### Size Criteria 

|Requirement|Value                    |
|-----------|-------------------------|
|Rows       |**70,000** (NLT 50,000 )|
|Columns    |**12** (NLT 10 )        |

### Columns

|# |Column     |Type |Description                                           |
|--|-----------|-----|------------------------------------------------------|
|1 |id         |int  |Patient ID                                            |
|2 |age        |int  |Age in **days** (not years — requires transformation) |
|3 |gender     |int  |1=Female, 2=Male                                      |
|4 |height     |int  |Height in cm                                          |
|5 |weight     |float|Weight in kg                                          |
|6 |ap_hi      |int  |Systolic blood pressure                               |
|7 |ap_lo      |int  |Diastolic blood pressure                              |
|8 |cholesterol|int  |1=Normal, 2=Above Normal, 3=Well Above Normal         |
|9 |gluc       |int  |Glucose: 1=Normal, 2=Above Normal, 3=Well Above Normal|
|10|smoke      |int  |Smoker: 0/1                                           |
|11|alco       |int  |Alcohol intake: 0/1                                   |
|12|active     |int  |Physical activity: 0/1                                |
|13|cardio     |int  |**Target** — CVD present: 0/1                         |

### Why This Dataset is NOT Clean 

1. **Blood pressure errors** — `ap_hi` contains values like -150, 1000+, and `ap_lo` > `ap_hi` (impossible clinically)
1. **Height outliers** — values as low as 55 cm and as high as 250 cm for adults
1. **Weight outliers** — values of 10 kg and 200+ kg present
1. **Age in days** — not directly usable; must be converted to years
1. **No pre-engineered features** — BMI, pulse pressure, hypertension flags all need to be derived
1. **Data from three different examination procedures** — inconsistencies exist

-----

## Project Structure

```
project/
├── cardiovascular_disease_project.ipynb   ← Full notebook (Steps 1–6)
├── app.py                                  ← Flask deployment
├── model.pkl                               ← Saved best model
├── scaler.pkl                              ← Saved StandardScaler
├── features.pkl                            ← Saved feature list
├── cardio_train.csv                        ← Dataset (download from Kaggle)
├── requirements.txt                        ← Python dependencies
└── README.md                               ← This file
```

-----

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Go to: <https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>  
Download `cardio_train.csv` and place it in the project folder.

### 3. Run the notebook

Open `cardiovascular_disease_project.ipynb` in Jupyter Notebook or VS Code.  
Run all cells top to bottom. This will generate:

- All 11 plots
- `model.pkl`, `scaler.pkl`, `features.pkl`

### 4. Launch the web app

```bash
python app.py
```

Open <http://127.0.0.1:5000> in your browser.

-----

## Project Criteria Checklist

|Criterion                            |Status                                                  |
|-------------------------------------|--------------------------------------------------------|
|Code runs without errors             |                                                       |
|Uses functions (no repetition)       | `evaluate_model()` function                           |
|Descriptive comments & variable names|                                                      |
|Research question clearly posed      |3 questions stated                                    |
|Data cleaning documented             |Step 2 — outliers, duplicates                         |
|≥ 6 variables investigated           |10 features explored                                  |
|Univariate + bivariate analysis      |                                                       |
|≥ 5 plot types                       |Bar, Histogram, Boxplot, Heatmap, Scatter, Violin, Pie|
|Feature engineering (new features)   |BMI, age_years, pulse_pressure, hypertension, obese   |
|Feature selection                    |SelectKBest (Filter Method)                           |
|≥ 3 algorithms compared              |Logistic Regression, Random Forest, Gradient Boosting |
|Parameter tuning discussed           |GridSearchCV with explanation                         |
|≥ 2 evaluation metrics               |Precision, Recall, F1, Accuracy, ROC-AUC              |
|Validation discussed                 |Train/test split + 5-fold CV                          |
|Precision ≥ 0.3, Recall ≥ 0.3        |Both ~0.72+                                           |
|Web app deployment                   |Flask app with live predictions                       |

-----

## Web App Screenshot

The web app features:

- Patient form (age, gender, height, weight, BP, cholesterol, glucose, lifestyle)
- Real-time CVD risk prediction
- Animated probability bar
- High/Low risk classification with visual feedback

-----
