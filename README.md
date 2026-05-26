# Medical Insurance Cost Prediction using Machine Learning

A comprehensive Data Science and Machine Learning project that explores, cleans, and builds a predictive model for medical insurance costs based on individual health and demographic attributes.

## 🚀 Project Overview
This repository contains a complete end-to-end workflow—from Exploratory Data Analysis (EDA) to Model Evaluation. The primary objective is to accurately predict the insurance `charges` feature using a Linear Regression model. 

### Key Highlights:
* **Data Preprocessing:** Handled duplicate records, encoded categorical features (One-Hot Encoding), and standardized continuous numerical variables.
* **Statistical Feature Engineering:** Implemented **Pearson Correlation** for numerical features and **Chi-Square Contingency Tests** for categorical features to drop statistically insignificant predictors.
* **Performance:** Achieved an Adjusted $R^2$ score of **~0.799** (approx. 80%) on the test set.

---

## 📊 Dataset Description
The dataset (`insurance.csv`) consists of 1,338 rows and 7 columns:
* `age`: Age of primary beneficiary
* `sex`: Insurance contractor gender (female, male)
* `bmi`: Body mass index ($kg / m^2$)
* `children`: Number of children covered by health insurance / Number of dependents
* `smoker`: Smoking habits (yes, no)
* `region`: The beneficiary's residential area in the US (northeast, southeast, northwest, southwest)
* `charges`: Individual medical costs billed by health insurance (**Target Variable**)

---

## 🛠️ Pipeline Architecture

1. **Exploratory Data Analysis (EDA):** Visualized feature distributions using histograms, countplots, and boxplots to detect structural trends and outliers.
2. **Data Cleaning:** Removed duplicates, mapped binary classes (`is_female`, `is_smoker`), and one-hot encoded geographic zones.
3. **Feature Selection:** * Used Pearson Correlation coefficients to understand linear relationships with `charges`.
   * Applied `chi2_contingency` on binned target variables to drop weak indicators like `region_southwest` and `region_northwest`.
4. **Model Training:** Split data into an 80/20 train-test split and trained a Scikit-Learn `LinearRegression` model on the finalized features.

---

## 📈 Model Evaluation
The final linear model was evaluated using the Adjusted Coefficient of Determination:

* **Adjusted $R^2$ Score:** `0.79879` (~80%)

This indicates that approximately 80% of the variance in insurance medical costs can be explained by the selected features (`age`, `is_female`, `bmi`, `children`, `is_smoker`, `region_southeast`, and `bmi_category_Obese`).

---

