# Import necessary libraries
import pandas as pd
import numpy as np
import json
import re

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Step 1: Load the data
print("Loading train and test datasets...")
train = pd.read_csv('train.csv', encoding='utf-8')
test = pd.read_csv('test.csv', encoding='utf-8')
print("Datasets loaded successfully.\n")

# Step 2: Exploratory Data Analysis (EDA) without plots
print("Starting Exploratory Data Analysis (EDA)...\n")

# Display basic information about the training set
print("Training Data Information:")
print(train.info())
print("\n")

# Display the first few rows of the training set
print("First 5 rows of training data:")
print(train.head())
print("\n")

# Check for missing values in training data
print("Missing Values in Training Data:")
print(train.isnull().sum())
print("\n")

# Display basic statistics of numerical features
print("Statistical Summary of Numerical Features:")
print(train.describe())
print("\n")

# Display value counts for categorical features
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for feature in categorical_features:
    print(f"Value counts for {feature}:")
    print(train[feature].value_counts())
    print("\n")

# Check correlation with target variable
print("Correlation of Numerical Features with Target:")
print(train.corr()['Transported'].sort_values(ascending=False))
print("\n")

print("EDA completed.\n")

# Step 3: Data Preprocessing and Feature Engineering
print("Starting Data Preprocessing and Feature Engineering...\n")

def preprocess_data(df, is_train=True):
    # Extract group and passenger number from PassengerId
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Passenger_Number'] = df['PassengerId'].apply(lambda x: int(x.split('_')[1]))
    
    # Extract Deck, Num, and Side from Cabin
    def extract_cabin(cabin):
        if pd.isnull(cabin):
            return ['Unknown', np.nan, 'Unknown']
        parts = cabin.split('/')
        deck = parts[0] if len(parts) > 0 else 'Unknown'
        num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else np.nan
        side = parts[2] if len(parts) > 2 else 'Unknown'
        return [deck, num, side]
    
    df[['Deck', 'Cabin_Num', 'Side']] = df['Cabin'].apply(lambda x: pd.Series(extract_cabin(x)))
    
    # Extract Title from Name
    df['Title'] = df['Name'].apply(lambda x: re.findall(r',\s*([^\.]*)\.', x)[0] if pd.notnull(x) else 'Unknown')
    
    # Calculate total spending
    spending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['Total_Spending'] = df[spending_features].sum(axis=1)
    
    # Handle missing values
    # For numerical features, fill missing with median
    numerical_features = ['Age', 'Cabin_Num', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Spending']
    for feature in numerical_features:
        imputer = SimpleImputer(strategy='median')
        df[feature] = imputer.fit_transform(df[[feature]])
    
    # For categorical features, fill missing with mode
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['Side'] = df['Side'].fillna('Unknown')
    df['Title'] = df['Title'].fillna('Unknown')
    df['VIP'] = df['VIP'].fillna(False)
    
    # Encode categorical variables
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Side', 'Deck', 'Title', 'VIP', 'Group']
    
    # Initialize LabelEncoder for binary categorical features
    le = LabelEncoder()
    for col in ['CryoSleep', 'VIP', 'Transported'] if is_train else ['CryoSleep', 'VIP']:
        df[col] = le.fit_transform(df[col])
    
    # One-Hot Encoding for other categorical features
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'Side', 'Deck', 'Title', 'Group'], drop_first=True)
    
    return df

# Preprocess train and test data
train_processed = preprocess_data(train, is_train=True)
test_processed = preprocess_data(test, is_train=False)

# Drop unnecessary columns
drop_features = ['PassengerId', 'Name', 'Cabin']
train_processed.drop(columns=drop_features, inplace=True)
test_ids = test['PassengerId']  # Save PassengerId for submission
test_processed.drop(columns=drop_features, inplace=True)

print("Data Preprocessing and Feature Engineering completed.\n")

# Step 4: Feature Selection and Model Training
print("Starting Feature Selection and Model Training...\n")

# Define features and target
X = train_processed.drop('Transported', axis=1)
y = train_processed['Transported']

# Split the data into training and validation sets for evaluation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
print("Performing cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

# Print cross-validation scores
for idx, score in enumerate(cv_scores):
    print(f"Fold {idx + 1} Accuracy: {score:.4f}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}\n")

# Train the model on the entire training data
print("Training the model on the entire training data...")
model.fit(X_train, y_train)
print("Model training completed.\n")

# Step 5: Evaluate the model's performance on the validation set
print("Evaluating model performance on the validation set...")
valid_predictions = model.predict(X_valid)
validation_accuracy = (valid_predictions == y_valid).mean()
print(f"Validation Accuracy: {validation_accuracy:.4f}\n")

# Step 6: Save progress report to progress_report.json
print("Saving progress report to progress_report.json...")
progress_report = {
    'cross_validation_scores': cv_scores.tolist(),
    'mean_cross_validation_accuracy': cv_scores.mean(),
    'validation_accuracy': validation_accuracy
}

with open('progress_report.json', 'w', encoding='utf-8') as f:
    json.dump(progress_report, f, ensure_ascii=False, indent=4)
print("Progress report saved successfully.\n")

# Step 7: Make predictions on the test set and prepare submission file
print("Making predictions on the test set...")
test_predictions = model.predict(test_processed)

# Prepare submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': test_predictions
})

# Ensure 'Transported' is boolean
submission['Transported'] = submission['Transported'].astype(bool)

# Save submission to CSV
print("Saving submission to submission.csv...")
submission.to_csv('submission.csv', index=False, encoding='utf-8')
print("Submission file saved successfully.\n")

print("All steps completed successfully.")

