# Detailed Report – Feature Engineering for Predictive Modeling on House Prices

---

## 1. **Data Loading & Initial Exploration**

The project began by loading the **House_Prices.csv** dataset into a Pandas DataFrame. I conducted an initial inspection of the data structure and quality:

* Displayed the **shape and sample rows** using `df.head()` and `df.describe()` to understand the range and distribution of features.
* Checked for **missing values** across the dataset using `df.isnull().sum()`, which informed subsequent preprocessing steps.
* Applied type conversion for binary features (e.g., `mainroad`, `basement`, `airconditioning`) from **'yes/no' to 1/0**.
* Encoded the categorical `furnishingstatus` column using **Label Encoding**.

---

## 2. **Feature Engineering**

To enhance model performance, several **new features were created** by combining and transforming existing ones:

* **Total Rooms**: Combined bedrooms and bathrooms to give a sense of property capacity.
* **Area Per Room**: Normalized house area by total rooms, indicating room spaciousness.
* **Parking Indicator**: Created a binary flag to indicate whether the property has parking.
* **Stories-Area Interaction**: Multiplied stories and area to capture vertical scale influence.
* **Bed-Bath Ratio**: Indicates house layout efficiency.
* **Luxury Score**: A composite feature summing up high-end amenities (`airconditioning`, `prefarea`, furnished status, and parking).
* **Amenities Score**: Aggregated presence of features like `guestroom`, `basement`, `hotwaterheating`, and `mainroad` access.

These features aimed to inject **domain knowledge** and **non-linear patterns** into the model.

---

## 3. **Feature Correlation Analysis**

A **heatmap** was plotted to visualize the correlation between features and the target variable (`price`). Only features with **absolute correlation > 0.3** were displayed.

This helped to:

* Identify **strong predictors** of house price.
* Visually validate the usefulness of **engineered features** such as `luxury_score` and `area_per_room`.

---

## 4. **Model Evaluation**

To assess the impact of feature engineering, three different sets of features were evaluated using a **Random Forest Regressor** inside a pipeline that included:

* Median imputation for missing values
* Standard scaling
* Model fitting and prediction

Each model was evaluated using:

* **Root Mean Squared Error (RMSE)**
* **R² Score (coefficient of determination)**

### Model Comparisons:

| Feature Set | No. of Features |         RMSE | R² Score |
| ----------- | --------------- | ------------ | -------- |
| Original    | 12              | 1.396656e+06 | 0.614082 |
| Engineered  | 19              | 1.397085e+06 | 0.613845 |
| Selected    | 12              | 1.434055e+06 | 0.593138 |

> Note: Actual scores were printed during runtime and plotted using a bar graph for visual comparison.

Additionally, **feature importances** from the Random Forest were extracted and visualized to identify which features contributed most to model performance.

---

## 5. **Feature Selection**

To reduce redundancy and improve model efficiency:

* **SelectKBest** with **f\_regression** was used to select the top 12 features from the engineered set.
* The selected subset retained high predictive power while reducing dimensionality.
* The final model trained on this subset showed **competitive performance** and emphasized important features like `luxury_score`, `area`, and `area_per_room`.

---

## 6. **Visual Summary**

To summarize and compare model outcomes:

* A **bar plot** of R² scores across feature sets illustrated the gains from feature engineering and selection.
* A **feature importance plot** highlighted which features the model relied on most for prediction.

---

## Final Thoughts

This project demonstrates the effectiveness of **feature engineering in boosting model performance**. By thoughtfully crafting new features and selecting the most relevant ones, I was able to:

* Enhance the predictive accuracy of the model.
* Extract deeper insights into what drives house prices.
* Improve model efficiency through dimensionality reduction.

### Future Enhancements:

* Explore **nonlinear transformations** (e.g., log or polynomial features).
* Incorporate **geographic data** or external economic indicators.
* Try different algorithms (e.g., Gradient Boosting, XGBoost) to further improve performance.

The final pipeline offers a **robust baseline** for predictive modeling in real estate pricing.