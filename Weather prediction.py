import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv(r'C:\Users\Rishab\Downloads\weather.csv')

print("Weather Dataset:\n")
print(df.head())


# Handle Missing Values

df = df.dropna()

# Convert Categorical Columns

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

df['WindGustDir'] = le1.fit_transform(df['WindGustDir'])
df['WindDir9am'] = le2.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = le3.fit_transform(df['WindDir3pm'])
df['RainToday'] = le4.fit_transform(df['RainToday'])


# Target Column Encoding

target = LabelEncoder()
df['RainTomorrow'] = target.fit_transform(df['RainTomorrow'])


# Features and Target

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']


# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Model

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)


# Prediction

y_pred = model.predict(X_test)


# Accuracy

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Actual vs Predicted Values

comparison = pd.DataFrame({
    'Actual': target.inverse_transform(y_test),
    'Predicted': target.inverse_transform(y_pred)
})

print("\nActual vs Predicted:\n")
print(comparison.head(20))


# Actual vs Predicted Graph

plt.figure(figsize=(10, 5))

plt.plot(comparison['Actual'].replace({'No':0, 'Yes':1}).values[:50],
         label='Actual', marker='o')

plt.plot(comparison['Predicted'].replace({'No':0, 'Yes':1}).values[:50],
         label='Predicted', marker='x')

plt.title("Actual vs Predicted Rain Values")

plt.xlabel("Sample Index")

plt.ylabel("Rain (0 = No, 1 = Yes)")

plt.legend()

plt.show()


# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# Predict New Data

new_data = [[
    10.0, 25.0, 0.0, 5.0, 8.0,
    1, 35.0, 2, 3, 10.0,
    15, 60, 40, 1015.0, 1012.0,
    4, 5, 18.0, 24.0, 0, 0.0
]]

prediction = model.predict(new_data)

# Probability Prediction
probability = model.predict_proba(new_data)

rain_probability = probability[0][1] * 100
no_rain_probability = probability[0][0] * 100

result = target.inverse_transform(prediction)

print("\nRain Tomorrow Prediction:", result[0])
print(f"\nChance of Rain: {rain_probability:.2f}%")
print(f"Chance of No Rain: {no_rain_probability:.2f}%")


# Count Plot

sns.countplot(x=target.inverse_transform(df['RainTomorrow']))
plt.title("Rain Tomorrow Count")
plt.show()


# Heatmap

plt.figure(figsize=(12, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')

plt.title("Weather Dataset Heatmap")
plt.show()