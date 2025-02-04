import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Check class balance for target variable ('LeaveOrNot' is the target variable)
if 'LeaveOrNot' in df.columns:
    print("\nTarget Variable Distribution:")
    print(df['LeaveOrNot'].value_counts(normalize=True))
    sns.countplot(x='LeaveOrNot', data=df)
    plt.title("Distribution of Employees Leaving")
    plt.show()

# Correlation heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Categorical variable analysis
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Features:", categorical_features)
for col in categorical_features:
    plt.figure(figsize=(10, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()

# Distribution of numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Features:", numerical_features)
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# Box plots to check for outliers
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Box Plot of {col}")
    plt.show()

# Relationship between features and target variable (for classification tasks)
if 'LeaveOrNot' in df.columns:
    for col in numerical_features:
        if col != 'LeaveOrNot':
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='LeaveOrNot', y=col, data=df)
            plt.title(f"{col} vs. LeaveOrNot")
            plt.show()

#----------------------------------------------------------------------------#

# Explainable Boosting Machine (EBM) Analysis
if 'LeaveOrNot' in df.columns:
    from sklearn.model_selection import train_test_split
    from interpret.glassbox import ExplainableBoostingClassifier
    from interpret import show

    # Prepare the data: Separate features and target variable
    target = 'LeaveOrNot'
    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical columns to the 'category' datatype (if not already)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Explainable Boosting Machine
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X_train, y_train)

    # Global Explanation: This shows how each feature contributes to the model's prediction overall.
    ebm_global = ebm.explain_global(name='EBM Global Explanation')

    # Interactive view
    show(ebm_global)

    # Local Explanation: Explain a single prediction from the test set.
    sample_index = 0  # Change the index to view different examples
    ebm_local = ebm.explain_local(X_test.iloc[[sample_index]], y_test.iloc[[sample_index]])
    show(ebm_local)
