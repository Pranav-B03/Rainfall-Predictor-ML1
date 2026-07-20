# 🌧️ Predicting Rainfall with a Random Forest

A machine learning project that predicts whether it will rain on a given day in districts of **Uttar Pradesh, India**, using historical daily weather data and a **Random Forest Classifier**.

---

## 1. Project Overview

This project builds a **binary classification model** that predicts whether rainfall will occur (**Rain** or **No Rain**) on a given day, for a given location in Uttar Pradesh, based on daily meteorological measurements such as humidity, temperature, pressure, wind speed, and UV index.

The project takes raw weather data through a complete machine learning pipeline — from data loading and cleaning, through exploratory data analysis (EDA), class balancing, model training, and evaluation — ending with a working model that can predict rainfall for a new, unseen day.

---

## 2. Problem Statement

Rainfall is one of the most important — and hardest to predict — weather events, especially in agriculture-dependent regions like Uttar Pradesh. Accurate short-term rainfall prediction can help:

- Farmers plan irrigation, sowing, and harvesting
- Local authorities prepare for flooding or water shortages
- Citizens plan daily activities around expected weather

This project addresses the problem: **"Given a day's atmospheric conditions (humidity, temperature, pressure, wind, etc.) at a specific location, can we predict whether it will rain?"**

The target is framed as a **binary classification problem**:
- `1` → Rain occurred (precipitation > 0)
- `0` → No rain occurred (precipitation = 0)

---

## 3. Project Objectives

- Analyze a large historical weather dataset (2005–2025) covering districts of Uttar Pradesh
- Clean and prepare the raw data for machine learning
- Engineer a binary rainfall target from the raw precipitation values
- Handle class imbalance in the target variable
- Train a Random Forest classifier to predict rainfall occurrence
- Evaluate the model using standard classification metrics
- Demonstrate the trained model on a new, unseen input (mock day prediction)

---

## 4. Dataset Information

| Detail | Value |
|---|---|
| File name | `UP_rainfall_dataset.csv` |
| Total records | **565,210** rows |
| Total columns | **20** columns |
| Missing values | **None** (0 missing values in any column) |
| Districts covered | **71** districts of Uttar Pradesh |
| Time range | Year **2005 – 2025** (daily records: `YEAR`, `MO`, `DY`) |

### Column Description

| Column | Description |
|---|---|
| `YEAR`, `MO`, `DY` | Year, Month, Day of the observation |
| `RH2M` | Relative Humidity at 2 meters (%) |
| `T2MDEW` | Dew Point Temperature at 2 meters (°C) |
| `QV2M` | Specific Humidity at 2 meters |
| `PS` | Surface Pressure |
| `WS50M` | Wind Speed at 50 meters |
| `PRECTOTCORR` | Corrected Precipitation (used to derive the target variable) |
| `T2MWET` | Wet Bulb Temperature at 2 meters |
| `WD50M` | Wind Direction at 50 meters |
| `T2M_MAX` | Maximum Temperature at 2 meters |
| `T2M_MIN` | Minimum Temperature at 2 meters |
| `ALLSKY_SFC_UV_INDEX` | All-Sky Surface UV Index |
| `TS` | Earth Skin Temperature |
| `PSC` | Corrected Surface Pressure |
| `WSC` | Corrected Wind Speed |
| `DISTRICT` | Name of the district (categorical) |
| `LATITUDE`, `LONGITUDE` | Geographic coordinates of the district |

> The column naming pattern (`RH2M`, `T2MDEW`, `WS50M`, `ALLSKY_SFC_UV_INDEX`, etc.) follows common satellite/reanalysis-based meteorological parameter naming conventions.

The **target variable**, `rainfall`, is **engineered** from `PRECTOTCORR`:
```python
data['rainfall'] = data['PRECTOTCORR'].apply(lambda x: 1 if x > 0 else 0)
```

---

## 5. Technologies and Libraries Used

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (`RandomForestClassifier`, `train_test_split`, `resample`) |
| Evaluation | `scikit-learn.metrics` (`accuracy_score`, `classification_report`, `confusion_matrix`) |
| Model Serialization | `pickle` (imported, available for saving/loading models) |
| Environment | Jupyter Notebook |

---

## 6. Project Workflow

The notebook follows a clear, linear machine learning pipeline:

```
1. Import Libraries
2. Load Dataset (UP_rainfall_dataset.csv)
3. Exploratory Data Analysis (EDA)
4. Data Cleaning (DISTRICT column standardization)
5. Feature Engineering (create binary 'rainfall' target)
6. Visualize Class Distribution
7. Feature Selection (drop leakage & categorical columns)
8. Handle Class Imbalance (downsampling)
9. Train-Test Split (80/20)
10. Train Random Forest Classifier
11. Evaluate Model (accuracy, classification report, confusion matrix)
12. Visualize Feature Importance
13. Test Model on a New (Mock) Observation
```

---

## 7. Data Preprocessing

The following preprocessing steps were applied to the raw dataset:

1. **Missing Value Check** — confirmed there were **zero missing values** across all 20 columns, so no imputation was required.
2. **DISTRICT Column Cleaning** — the raw `DISTRICT` column had inconsistent text formatting (e.g., `"amroha"`, `"bahraich"`, `"bhadohi "` with trailing spaces, mixed casing). This was cleaned using:
   ```python
   data['DISTRICT'] = data['DISTRICT'].str.strip()   # remove leading/trailing whitespace
   data['DISTRICT'] = data['DISTRICT'].str.title()    # standardize capitalization
   ```
3. **Target Variable Creation** — a new binary column `rainfall` was engineered from `PRECTOTCORR` (1 if precipitation > 0, else 0).
4. **Feature/Target Separation & Leakage Removal** — `PRECTOTCORR` was dropped from the feature set because it directly encodes the answer (data leakage), and `DISTRICT` was dropped because it is a non-numeric categorical column not used by the model:
   ```python
   X_features = data.drop(columns=['PRECTOTCORR', 'DISTRICT', 'rainfall'])
   y_target = data['rainfall']
   ```
5. **Class Imbalance Handling** — the target classes were imbalanced (No Rain: 301,001 vs Rain: 264,209). The **majority class was downsampled** (without replacement) to match the minority class size, producing a perfectly balanced dataset of 264,209 samples per class.

---

## 8. Exploratory Data Analysis (EDA)

The notebook performs several EDA steps to understand the dataset before modeling:

- **Dataset structure check** — shape, column names, data types (`data.info()`), and statistical summary (`data.describe()`)
- **Missing value inspection** — confirmed a clean dataset with no nulls
- **Rainfall (`PRECTOTCORR`) distribution** — summary statistics show most days have low/zero precipitation, with a right-skewed distribution up to a maximum of 279.27 mm
- **Unique district inspection** — checked before and after text cleaning
- **Geographical scatter plot** — plots the latitude/longitude of all 71 districts to visualize their spatial distribution across Uttar Pradesh
- **Rainfall class distribution (pie chart)** — visualizes the percentage split between "Rain" and "No Rain" days before balancing, confirming a moderate class imbalance (~53% No Rain vs ~47% Rain)
- **Feature importance plot** — a horizontal bar chart (Viridis color scale) showing the relative importance each feature had in the trained Random Forest's decisions. Humidity and moisture-related features (such as `RH2M`, `T2MDEW`, and `QV2M`) are visibly among the most influential predictors, while pure calendar identifiers like `YEAR` contribute comparatively less.

---

## 9. Features Used

After removing the leakage column (`PRECTOTCORR`) and the categorical column (`DISTRICT`), the model was trained on **18 numeric features**:

```
YEAR, MO, DY, RH2M, T2MDEW, QV2M, PS, WS50M, T2MWET,
WD50M, T2M_MAX, T2M_MIN, ALLSKY_SFC_UV_INDEX, TS, PSC,
WSC, LATITUDE, LONGITUDE
```

These cover four broad categories:
- **Temporal**: `YEAR`, `MO`, `DY`
- **Humidity/Moisture**: `RH2M`, `T2MDEW`, `QV2M`, `T2MWET`
- **Temperature/Pressure**: `T2M_MAX`, `T2M_MIN`, `PS`, `PSC`, `TS`
- **Wind & Radiation**: `WS50M`, `WD50M`, `WSC`, `ALLSKY_SFC_UV_INDEX`
- **Location**: `LATITUDE`, `LONGITUDE`

---

## 10. Machine Learning Models Used

| Model | Library | Notes |
|---|---|---|
| **Random Forest Classifier** | `sklearn.ensemble.RandomForestClassifier` | Trained with default hyperparameters and `random_state=42` for reproducibility |

Only one model — a Random Forest Classifier — is trained and evaluated in this project. `GridSearchCV` and `cross_val_score` are imported at the top of the notebook but are **not actually used** in the current workflow (no hyperparameter tuning or cross-validation was performed).

---

## 11. Model Training and Testing

- **Balanced dataset size:** 528,418 samples (264,209 per class)
- **Train-Test Split:** 80% training / 20% testing, using `train_test_split(..., test_size=0.2, random_state=42)`

| Set | Shape |
|---|---|
| Training features (`X_train`) | (422,734, 18) |
| Testing features (`X_test`) | (105,684, 18) |

- **Model initialization and training:**
  ```python
  rf_model = RandomForestClassifier(random_state=42)
  rf_model.fit(X_train, y_train)
  ```
- **Prediction on test set:**
  ```python
  y_pred = rf_model.predict(X_test)
  ```

---

## 12. Evaluation Metrics

The model was evaluated using three standard classification metrics:

- **Accuracy Score** — overall percentage of correct predictions
- **Classification Report** — precision, recall, and F1-score per class
- **Confusion Matrix** — breakdown of true/false positives and negatives

---

## 13. Results and Performance

### Overall Accuracy
```
Model Accuracy: 94.79%
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (No Rain) | 0.94 | 0.96 | 0.95 | 52,908 |
| 1 (Rain) | 0.95 | 0.94 | 0.95 | 52,776 |
| **Accuracy** | | | **0.95** | 105,684 |
| **Macro Avg** | 0.95 | 0.95 | 0.95 | 105,684 |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 105,684 |

### Confusion Matrix

|  | Predicted: No Rain | Predicted: Rain |
|---|---|---|
| **Actual: No Rain** | 50,569 | 2,339 |
| **Actual: Rain** | 3,164 | 49,612 |

**Interpretation:** The model correctly classified the vast majority of both rain and no-rain days, with balanced precision and recall (~94–96%) for both classes — indicating the model is not biased toward either class, largely thanks to the downsampling step used to balance the dataset before training.

---

## 14. Project Structure

```
rainfall-prediction/
│
├── FINAL_VERSION.ipynb        # Main Jupyter Notebook (full ML pipeline)
├── UP_rainfall_dataset.csv    # Dataset (daily weather records, UP districts)
└── README.md                  # Project documentation (this file)
```

---

## 15. How to Run the Project

1. **Clone or download** the project folder containing `FINAL_VERSION.ipynb` and `UP_rainfall_dataset.csv`.
2. Make sure both files are in the **same directory**.
3. Install the required libraries (see [Installation Requirements](#16-installation-requirements)).
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `FINAL_VERSION.ipynb` and run all cells from top to bottom (`Kernel → Restart & Run All`).

---

## 16. Installation Requirements

This project requires **Python 3.x** along with the following libraries:

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install them all with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

`pickle` is part of Python's standard library and does not need separate installation.

---

## 17. Example Usage

Once the model (`rf_model`) is trained, it can be used to predict rainfall for a new day by passing in the same 18 features it was trained on:

```python
# 1. Create a dictionary of weather measurements for a single day
mock_day_features = {
    'YEAR': [2026],
    'MO': [7],          # July (Monsoon season)
    'DY': [15],
    'RH2M': [88.50],    # High humidity
    'T2MDEW': [24.10],
    'QV2M': [18.50],
    'PS': [99.40],
    'WS50M': [5.20],
    'T2MWET': [25.30],
    'WD50M': [145.0],
    'T2M_MAX': [32.50],
    'T2M_MIN': [24.00],
    'ALLSKY_SFC_UV_INDEX': [0.35],
    'TS': [26.10],
    'PSC': [90.20],
    'WSC': [4.80],
    'LATITUDE': [25.31],
    'LONGITUDE': [83.01]
}

# 2. Convert to a DataFrame
input_df = pd.DataFrame(mock_day_features)

# 3. Predict
predicted_class = rf_model.predict(input_df)

# 4. Display result
if predicted_class[0] == 1:
    print("FINAL PREDICTION: [1] -> Precipitation Expected (Rainfall)")
else:
    print("FINAL PREDICTION: [0] -> Clear Conditions Expected (No Rainfall)")
```

**Output (from the notebook):**
```
FINAL PREDICTION: [ 1 ] -> Precipitation Expected (Rainfall)
```

---

## 18. Limitations

- **No hyperparameter tuning:** `GridSearchCV` and `cross_val_score` are imported but never used — the Random Forest runs with default `scikit-learn` hyperparameters.
- **No cross-validation:** performance is based on a single 80/20 train-test split rather than k-fold cross-validation, so results may vary slightly with different splits.
- **Downsampling discards data:** balancing the classes by downsampling the majority class means a large portion (~36,792 rows) of the original "No Rain" data was not used in training, which could discard potentially useful information.
- **Location dropped as a categorical feature:** the `DISTRICT` name itself is excluded (only `LATITUDE`/`LONGITUDE` are kept), so any district-specific patterns not captured by geographic coordinates are lost.
- **Binary target simplification:** the target only predicts *whether* it rains, not *how much* rainfall to expect (rainfall intensity/amount is not modeled).
- **No model persistence in the current run:** although `pickle` is imported, the trained model is not actually saved to disk in the notebook, so it must be retrained each time the notebook is run.

---

## 19. Future Improvements

- Perform **hyperparameter tuning** using `GridSearchCV` (already imported) to optimize parameters like `n_estimators`, `max_depth`, and `min_samples_split`.
- Apply **k-fold cross-validation** using `cross_val_score` (already imported) for a more robust performance estimate.
- Compare the Random Forest against other algorithms (e.g., Logistic Regression, XGBoost, Gradient Boosting) to benchmark performance.
- Explore alternative class-balancing techniques such as **SMOTE (oversampling)** instead of downsampling, to avoid discarding data.
- Extend the target from binary classification to a **regression problem** that predicts actual rainfall amount (mm).
- Save the trained model using `pickle` (or `joblib`) so it can be reused without retraining.
- Build a simple **web app or API** (e.g., with Flask/Streamlit) around the trained model for interactive predictions.
- Incorporate `DISTRICT` as an encoded categorical feature (e.g., one-hot encoding) to capture location-specific rainfall patterns beyond coordinates.

---

## 20. Conclusion

This project demonstrates a complete, end-to-end machine learning workflow for predicting daily rainfall occurrence across districts of Uttar Pradesh using a Random Forest Classifier. Starting from a large, real-world weather dataset of over 565,000 records, the project performs careful data cleaning, handles class imbalance through downsampling, and trains a model that achieves **~94.8% accuracy** with balanced precision and recall across both classes. The result is a reliable baseline model for binary rainfall prediction, along with a clear roadmap of improvements (hyperparameter tuning, cross-validation, and alternative balancing techniques) that could push performance even further.

---

*This README was generated based on a direct analysis of the `FINAL_VERSION.ipynb` notebook and its outputs.*
