import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
file_path = 'dataset/Employee.csv'
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Info:")
df.info()
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

#----------------------------------------------------------------------------#

# Decision Tree Classifier Analysis
if 'LeaveOrNot' in df.columns:
    # Prepare the data: Separate features and target variable
    target = 'LeaveOrNot'
    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical columns to the 'category' datatype (if not already)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree classifier
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)

    # Feature importance
    feature_importances = tree_clf.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

    # Evaluate the model on the test set
    y_pred = tree_clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Visualize the Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_clf, feature_names=features, class_names=['Stay', 'Leave'], filled=True, rounded=True)
    plt.title('Decision Tree Visualization')
    plt.show()
