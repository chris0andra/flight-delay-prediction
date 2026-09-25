# Predicting U.S. Flight Delays Using Machine Learning

## ADS 504 Final Team Project

**Team Members:** Christopher Andra, Caly Nguyen, Titus Sun

## Project Overview

Flight delays are a common problem in commercial aviation and can affect passengers, airlines, airports, crews, and connecting flight schedules. This project uses machine learning to predict whether a completed, non-diverted U.S. domestic flight will arrive at least 15 minutes late.

The project was designed around a realistic prediction setting. Only information available before departure was used as model input, while variables recorded during or after the flight were excluded to prevent data leakage.

The original analysis was completed as an ADS 504 team project using the **2015 Flight Delays and Cancellations** dataset. This repository contains the complete project notebook and will also serve as the starting point for continued model development.

## Research Questions

1. How effectively can significant arrival delays be predicted using only pre-departure information?
2. Which machine learning model performs best at identifying significantly delayed flights?
3. Do more complex models provide stronger delay detection than a simpler baseline model when evaluated on future-month flight data?

## Repository File

The complete original project is contained in one Jupyter Notebook:

`ADS504Projectflightdelay.ipynb`

The notebook includes data loading, data description, exploratory data analysis, cleaning, preprocessing, feature engineering, model training, evaluation, visualizations, and final test results.

## Dataset

The project uses the **2015 Flight Delays and Cancellations** dataset.

The original data include:

- `flights.csv` — 5,819,079 flight records and 31 variables
- `airports.csv` — airport reference information
- `airlines.csv` — airline reference information
- `T_MASTER_CORD.csv` — Bureau of Transportation Statistics airport reference data used to resolve the October airport identifier issue

Each row in the flights dataset represents one scheduled U.S. domestic flight during 2015.

### Data Access

The raw data files are not stored directly in this GitHub repository because of their size.

The project data files are available through Google Drive:

**[Project Data Files - Google Drive](https://drive.google.com/drive/folders/1N38b9B2Y5sB2QGI9R7MbGqrYKOGmiYOx?usp=sharing)**

The Google Drive folder is intended to be shared as **Anyone with the link - Viewer** so the files can be downloaded without allowing others to modify them.

## Target Variable

The target variable is `SIGNIFICANT_DELAY`.

A flight is classified as significantly delayed when:

```text
ARRIVAL_DELAY >= 15 minutes
```

Canceled and diverted flights were excluded from the modeling population because they did not have a normal completed-flight arrival-delay outcome.

After exclusions, approximately **5.71 million eligible flights** remained.

The target was imbalanced:

- **18.61%** significantly delayed
- **81.39%** not significantly delayed

Because of this imbalance, model evaluation focused on more than accuracy alone.

## Data Cleaning and Preprocessing

The project included several cleaning and preprocessing steps before modeling.

One major issue involved airport identifiers in October. Flights in the other months used standard three-letter airport codes, while October records used five-digit numeric airport identifiers. The project used Bureau of Transportation Statistics reference data to convert the October identifiers instead of removing the month.

Additional preprocessing included:

- Removing canceled and diverted flights
- Removing records without a usable arrival-delay outcome
- Creating the `SIGNIFICANT_DELAY` target
- Converting scheduled departure and arrival times from HHMM format
- Treating `2400` as midnight
- Removing variables that would introduce data leakage
- Median imputation for numeric variables
- Standardization of numeric variables
- Most-frequent imputation for categorical variables
- One-hot encoding of categorical variables
- Excluding high-cardinality variables from the final modeling pipeline when necessary for memory limitations

## Feature Engineering

Engineered features were created only from information available before departure.

Examples include:

- `IS_WEEKEND`
- `DEPARTURE_PERIOD`
- `ARRIVAL_PERIOD`
- `SCHEDULED_DEPARTURE_HOUR`
- `SCHEDULED_ARRIVAL_HOUR`
- `SCHEDULED_OVERNIGHT`
- `DISTANCE_GROUP`
- `SPEED_PROXY_MPH`
- `ROUTE` for exploratory analysis

The final modeling pipeline used **12 numeric predictors** and **4 categorical predictors**.

## Exploratory Data Analysis

Exploratory analysis was used to understand the data, identify quality problems, examine class balance, and explore delay patterns before modeling.

Key findings included:

- Morning departures generally had lower significant-delay rates.
- Delay rates increased through the afternoon and evening.
- Monthly significant-delay rates varied throughout the year, roughly from 12% to 24%.
- January, February, and June showed some of the higher monthly significant-delay rates.
- Numeric correlations with the target were generally weak.
- Scheduled departure hour had one of the stronger numeric relationships with the target at approximately 0.14.
- The October airport identifier issue affected 486,165 flight records and required a separate correction step.

## Train, Validation, and Test Strategy

A **time-based split** was used instead of a random split because the project was designed to predict future flights using earlier flight data.

| Time Period | Purpose |
| --- | --- |
| January-October | Training |
| November | Validation and model selection |
| December | Final test |

The complete January-October training data contained millions of records. Because of Google Colab memory and runtime limitations, a reproducible random sample of **500,000 training records** was used with `random_state=42`.

The November validation set and December test set were kept unchanged.

## Machine Learning Models

Four primary models were included in the final comparison.

### Logistic Regression

Logistic Regression was used as the baseline model.

Main settings:

```text
max_iter = 1000
class_weight = balanced
random_state = 42
```

Balanced class weights were used to account for the minority delayed-flight class.

### Random Forest

Random Forest was used as a nonlinear ensemble model.

Main settings:

```text
n_estimators = 100
max_depth = 15
class_weight = balanced
random_state = 42
```

### Weighted XGBoost

XGBoost was used as a gradient-boosting model with weighting for the minority class.

Main settings:

```text
n_estimators = 100
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 4.3526
random_state = 42
```

The `scale_pos_weight` value was calculated from the ratio of non-delayed to delayed flights in the training sample.

### Weighted Neural Network

A Multi-Layer Perceptron Neural Network was also evaluated.

Main settings:

```text
hidden_layer_sizes = (64, 32)
activation = relu
solver = adam
batch_size = 1024
learning_rate_init = 0.001
max_iter = 200
early_stopping = True
random_state = 42
```

The original unweighted neural network predicted almost entirely the majority class. A weighted version was therefore trained using balanced sample weights and used in the final comparison.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrices
- ROC curves
- Precision-recall curves

Because only about 18.61% of flights were significantly delayed, **recall, F1-score, and PR-AUC** were especially important when interpreting model performance.

## November Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.699 | 0.214 | **0.363** | **0.269** | 0.606 | 0.204 |
| Random Forest | 0.821 | 0.260 | 0.093 | 0.136 | 0.609 | 0.209 |
| Weighted XGBoost | 0.796 | 0.247 | 0.166 | 0.199 | 0.607 | **0.209** |
| Weighted Neural Network | 0.825 | 0.234 | 0.064 | 0.100 | 0.560 | 0.184 |

At the default classification thresholds, Logistic Regression produced the highest recall and F1-score on the November validation data.

Random Forest and XGBoost had slightly stronger PR-AUC values, showing that the comparison could change if probability thresholds and model parameters were tuned more extensively.

## December Test Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.682 | 0.248 | **0.269** | **0.258** | 0.575 | 0.242 |
| Random Forest | 0.761 | 0.260 | 0.086 | 0.130 | 0.579 | 0.246 |
| Weighted XGBoost | 0.721 | 0.249 | 0.176 | 0.206 | **0.584** | **0.248** |
| Weighted Neural Network | 0.784 | **0.305** | 0.036 | 0.064 | 0.558 | 0.242 |

The original project selected Logistic Regression because recall and F1-score were emphasized. However, XGBoost achieved the strongest December ROC-AUC and PR-AUC, which motivates further investigation rather than treating the original model selection as final.

## Project Expansion

The original course project is complete, but this repository will continue from the existing notebook.

The next stage of the project will focus on:

- Classification-threshold tuning using the November validation data
- Hyperparameter tuning, especially for Random Forest and XGBoost
- Re-evaluating model selection after threshold optimization
- Comparing tuned models against the original baseline results
- Improving reproducibility of the final December comparison
- Exploring additional pre-departure features
- Investigating historical airline, airport, and route delay information
- Preserving December as the final test period during model development

The original results above will remain as the baseline so future improvements can be compared directly against the first version of the project.

## Team Contributions

- **Christopher Andra** — Data Description and Exploratory Data Analysis
- **Caly Nguyen** — Data Preprocessing and Feature Engineering
- **Titus Sun** — Machine Learning Modeling and Evaluation

Continued post-course development of this repository is being completed by **Christopher Andra**.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- Jupyter Notebook
- Google Colab
- GitHub

## How to Run the Project

1. Clone or download this repository.
2. Download the project data from the Google Drive link in the **Data Access** section.
3. Place the required data files where the notebook can access them.
4. Open `ADS504Projectflightdelay.ipynb` in Jupyter Notebook or Google Colab.
5. Update the file paths near the beginning of the notebook if your data are stored in a different location.
6. Run the notebook cells in order.

The current notebook was originally developed in Google Colab and mounts Google Drive for data access, so local users may need to replace the Colab-specific file paths with local paths.

## Data Source

**2015 Flight Delays and Cancellations**  
U.S. Department of Transportation flight data distributed through Kaggle.

Additional airport reference information from the **Bureau of Transportation Statistics** was used to correct the October airport identifiers.

## Academic Context

This repository began as an ADS 504 team project. The original notebook is being preserved as the baseline version of the analysis while the project is expanded beyond the original course submission.

## Status

**Original project:** Complete  
**Project expansion:** In progress

