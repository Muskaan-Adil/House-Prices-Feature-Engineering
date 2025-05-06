# House Price Dataset – Feature Engineering for Predictive Modeling

## Project Overview

This project focuses on applying **feature engineering techniques** to a real estate dataset containing house attributes and prices. The primary goal is to improve the performance of a predictive model by creating new features, analyzing their impact, and selecting the most important ones.

This project highlights the value of transforming raw data into informative features that can significantly enhance the accuracy and interpretability of machine learning models.

---

## Key Features

* **Data Inspection**: Loaded and examined the dataset structure, explored distributions, and identified missing values.
* **Preprocessing**: Encoded categorical variables and converted binary text fields into numeric format for modeling.
* **Feature Engineering**: Created new features such as room ratios, amenity scores, and interaction terms to capture complex relationships.
* **Feature Correlation Analysis**: Visualized and interpreted correlations between input features and house price using heatmaps.
* **Model Evaluation**: Compared performance across models trained with original, engineered, and selected features using RMSE and R².
* **Feature Selection**: Applied statistical feature selection to reduce dimensionality and retain high-impact predictors.
* **Visualization**: Created visual summaries of model performance and feature importance using bar plots and heatmaps.

---

## Dataset Information

**Source**: [Housing dataset containing various attributes of residential properties including area, number of rooms, amenities, and price](https://drive.google.com/file/d/1U_C-gWb6VwkSHAkru6tzAiufLJLY009B/view).

**Columns Include**:

* `area`: Square footage of the house.
* `bedrooms`, `bathrooms`: Number of rooms.
* `stories`: Number of floors.
* `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`: Binary indicators for amenities.
* `furnishingstatus`: Categorical variable indicating furnishing level.
* `parking`: Number of parking spaces.
* `price`: Target variable – house price in currency units.

---

## Feature Engineering Steps

### Loading and Initial Processing

* Loaded the dataset using `pandas` and explored it with `df.head()`, `df.describe()`, and `df.info()`.
* Converted binary "yes/no" columns into 0/1 format.
* Encoded `furnishingstatus` using `LabelEncoder`.

### Engineered Features

* `total_rooms`: Sum of bedrooms and bathrooms.
* `area_per_room`: Area divided by total rooms.
* `has_parking`: Binary flag for presence of parking.
* `stories_area`: Interaction between stories and area.
* `bed_bath_ratio`: Bedroom to bathroom ratio.
* `luxury_score`: Composite metric for high-end features.
* `amenities_score`: Count of available amenities.

---

## Model Building and Evaluation

Three models were trained using a **Random Forest Regressor** inside a `Pipeline` with imputation and scaling:

1. **Original Features** – using only raw dataset columns.
2. **Engineered Features** – original plus newly created features.
3. **Selected Features** – top features selected using `SelectKBest` with f\_regression.

Each model was evaluated using:

* **RMSE** (Root Mean Squared Error)
* **R² Score**

Feature importances from the Random Forest were plotted to highlight key predictors.

---

## Visualizations

Created:

* **Heatmap** of feature correlations with `price`.
* **Bar plots** showing feature importances for different models.
* **Performance comparison** plot across feature sets based on R² scores.

---

## Report

For a comprehensive explanation of each step, including preprocessing, feature generation, model results, and visual insights, refer to the full analysis in **[House_Prices_Report.md](House_Prices_Report.md)**.
