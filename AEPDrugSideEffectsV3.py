#Author: Bharathi Athinarayanan

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Read the data from the original CSV file
df = pd.read_csv('cleandata.csv')

# Get the unique values in each column
age = df['Age'].unique()
sex = df['Sex'].unique()
indications = df['Indicated for'].unique()
medications = df['Prescribing Medication'].unique()
contraindications = df['Concmittant medications'].unique()
medical_history = df['Medical History'].unique()
common_side_effects = df['Common Side effects'].unique()
rare_side_effects = df['Rare Side effects'].unique()
adverse_event = df['Adverse event'].unique()

synthetic_data = {
    'Age': np.random.choice(age, 1000),
    'Sex': np.random.choice(sex, 1000),
    'Indicated for': np.random.choice(indications, 1000),
    'Prescribing Medication': np.random.choice(medications, 1000),
    'Concmittant medications': np.random.choice(contraindications, 1000),
    'Medical History': np.random.choice(medical_history, 1000),
    'Common Side effects': np.random.choice(common_side_effects, 1000),
    'Rare Side effects': np.random.choice(rare_side_effects, 1000),
    'Adverse event': np.random.choice(adverse_event, 1000)
}

df_synthetic = pd.DataFrame(synthetic_data)

# Write the DataFrame with synthetic data to a new CSV file
df_synthetic.to_csv('synthetic_drug_data.csv', index=False)

df = pd.read_csv('synthetic_drug_data.csv')

# Define a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Indicated for', 'Prescribing Medication', 'Concmittant medications', 'Medical History'])
    ])

# Apply the preprocessor to your data
X = preprocessor.fit_transform(df.drop(['Common Side effects', 'Rare Side effects', 'Adverse event'], axis=1))

# Save the preprocessor to disk
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Split the data into targets (y_common, y_rare, y_adverse)
y_common = df['Common Side effects']
y_rare = df['Rare Side effects']
y_adverse = df['Adverse event']

# Split the data into training and test sets
X_train, X_test, y_train_common, y_test_common = train_test_split(X, y_common, test_size=0.2, random_state=42)
X_train, X_test, y_train_rare, y_test_rare = train_test_split(X, y_rare, test_size=0.2, random_state=42)
X_train, X_test, y_train_adverse, y_test_adverse = train_test_split(X, y_adverse, test_size=0.2, random_state=42)

# Define a parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None],
}

# Create and train the random forest classifiers with GridSearchCV
clf_common = GridSearchCV(RandomForestClassifier(random_state=42), param_grid)
clf_common.fit(X_train, y_train_common)

clf_rare = GridSearchCV(RandomForestClassifier(random_state=42), param_grid)
clf_rare.fit(X_train, y_train_rare)

clf_adverse = GridSearchCV(RandomForestClassifier(random_state=42), param_grid)
clf_adverse.fit(X_train, y_train_adverse)

# Make predictions on the test set
y_pred_common = clf_common.predict(X_test)
y_pred_rare = clf_rare.predict(X_test)
y_pred_adverse = clf_adverse.predict(X_test)

print("Predictions for Common Side Effects: ", y_pred_common)
print("Predictions for Rare Side Effects: ", y_pred_rare)
print("Predictions for Adverse Events: ", y_pred_adverse)

print("Accuracy for Common Side Effects: ", accuracy_score(y_test_common,y_pred_common))
print("Accuracy for Rare Side Effects: ", accuracy_score(y_test_rare,y_pred_rare))
print("Accuracy for Adverse Events: ", accuracy_score(y_test_adverse,y_pred_adverse))

# Save the trained models as pickle files
with open('model_common.pkl', 'wb') as f:
    pickle.dump(clf_common.best_estimator_, f)

with open('model_rare.pkl', 'wb') as f:
    pickle.dump(clf_rare.best_estimator_, f)

with open('model_adverse.pkl', 'wb') as f:
    pickle.dump(clf_adverse.best_estimator_, f)


from lime import lime_tabular

# Create a LimeTabularExplainer
explainer = lime_tabular.LimeTabularExplainer(X_train)

# Get the first instance from the test set
instance = X_test[0]

# Explain the prediction
exp = explainer.explain_instance(instance, clf_common.predict_proba)

# Get the explanation as a list
explanation = exp.as_list()

# Print the explanation
for feature, contribution in explanation:
    print(f"The feature '{feature}' contributes {contribution} to the prediction.")

######################################
