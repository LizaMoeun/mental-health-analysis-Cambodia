# %% [markdown]
# Mental Health Analysis in Cambodia Using ML

# %%
# Importing libraries for data manipulation
import pandas as pd
import numpy as np

# For data transformation
from scipy.stats import zscore

# For data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# For splitting the dataset
from sklearn.model_selection import train_test_split

# For model evaluation
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report


# %%
# Load the mental health dataset 
mental_data = pd.read_csv(r"C:\Users\Hour San Computer\OneDrive\Documents\Lesson SE\Year 2\semeter 3\Machine Leraing\mental_health_cambodia_dataset.csv")
# Displaying the first few records
print("First 5 records:")
print(mental_data.head())

# %%
# data info
mental_data.info()

# %%
# Descriptive statistics
mental_data.describe().round(2)

# %%
# Counting class distribution
class_counts = mental_data['Risk'].value_counts()
print("\nClass Counts:")
print(class_counts)

# Percentages
class_percent = (class_counts / len(mental_data)) * 100
print("\nClass Percentages:")
print(class_percent)


# %%
#Class distribution
sns.countplot(data=mental_data, x='Risk')
plt.title("Mental Health Diagnosis Distribution in Cambodia")
plt.show()

# %%
# Boxplot for selected features
features = ['Age', 'Income_Level', 'Trauma']
for feature in features:
    sns.boxplot(x='Risk', y=feature, data=mental_data)
    plt.title(f"Boxplot for {feature}")
    plt.show()

# %%
#Drop missing values
mental_data.dropna(inplace=True)

# Drop ID (not useful)
mental_data = mental_data.drop(columns=['ID'], errors='ignore')

#Separte features and target
X = mental_data.drop(columns=['Risk'])
y = mental_data['Risk']

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X)

#Standardize numeric features
X_scaled = X_encoded.apply(zscore)

print("\nStandardized features:")
print(X_scaled.head())


# %%
#Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LogisticRegression

# Initialize the instance of the algorithm
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# Use the instance to train the model
logistic_model.fit(X_train, y_train)


# %%
from sklearn.metrics import precision_score, recall_score, f1_score

# Making predictions on the test set
test_predictions = logistic_model.predict(X_test)

# Computing and printing the performance metrics
print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Precision:", precision_score(y_test, test_predictions, average='weighted'))
print("Recall:", recall_score(y_test, test_predictions, average='weighted'))
print("F1 Score:", f1_score(y_test, test_predictions, average='weighted'))

# %%
from sklearn.metrics import confusion_matrix

# Visualizing the confusion matrix
cm = confusion_matrix(y_test, test_predictions, labels=logistic_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic_model.classes_)
plt.figure(figsize=(8, 6))
disp.plot()
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
print("Classification Report:")
print(classification_report(y_test, test_predictions))

# %%
import joblib

# Saving the trained logistic regression model
joblib.dump(logistic_model, 'mental_health_logistic_model.joblib')


# %%
# Load the model
loaded_model = joblib.load('mental_health_logistic_model.joblib')

# Calculate means and stds from training data
feature_means = X_train.mean()
feature_stds = X_train.std()

# Define new unseen data — make sure it's 26 values in correct order!
new_data = [24, 1, 0, 2, 0, 1, 0, 0, 1, 0, 0, 1, 
            1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]


# Standardize
standardized_new_data = (np.array(new_data) - feature_means.values) / feature_stds.values
reshaped_new_data = standardized_new_data.reshape(1, -1)

# Predict
prediction = loaded_model.predict(reshaped_new_data)
print("\nPrediction on New Data:")
print("The new data is predicted as class:", prediction[0])



