# Weather Prediction & Exploratory Data Analysis

This repository contains a machine learning and exploratory data analysis (EDA) project that predicts whether it will rain tomorrow based on historical weather observations. It preprocesses the data, trains a Random Forest Classifier, evaluates the model performance, and generates multiple visual insights.

## Tech Stack

* **Language:** Python
* **Data Libraries:** Pandas
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn

## Features

* **Data Preprocessing:** Handles missing values and automatically encodes categorical variables using label encoders.
* **Model Training:** Utilizes a Random Forest Classifier to learn from weather parameters.
* **Performance Evaluation:** Outputs accuracy scores, comprehensive classification reports, and a confusion matrix.
* **Data Visualization:**
  * Actual vs. Predicted comparison plot
  * Heatmap showing correlation between features
  * Distribution of target labels (rain tomorrow vs. no rain)
  * Confusion matrix heatmap
* **Inference Capability:** Predicts both class (Rain/No Rain) and probability for new sample data.

## Getting Started

## Prerequisites

Ensure you have Python installed. You can install the required packages using:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Dataset

The script expects a weather dataset file named `weather.csv` located at:
`C:\Users\Rishab\Downloads\weather.csv`

*(Note: You can update the path on line 11 of the script if your dataset is located elsewhere.)*

### Running the Script

Execute the Python script using:

```bash
python "Weather prediction.py"
```

