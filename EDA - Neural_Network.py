import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""

1. Education: The educational qualifications of employees, including degree, institution, and field of study.

2. Joining Year: The year each employee joined the company, indicating their length of service.

3. City: The location or city where each employee is based or works.

4. Payment Tier: Categorization of employees into different salary tiers.

5. Age: The age of each employee, providing demographic insights.

6. Gender: Gender identity of employees, promoting diversity analysis.

7. Ever Benched:  Indicates if an employee has ever been temporarily without assigned work.

8. Experience in Current Domain: The number of years of experience employees have in their current field.

9. Leave or Not: a target column

"""

# Load dataset
file_path = r"C:\Users\deniz\Desktop\Explanaible AI\XAI_Codes\Employee.csv"
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


# Age Group
#We have divided the ages into groups to make them more understandable

df["AgeGroup"] = pd.cut(df["Age"], bins=[20, 25, 35, 50], labels=["Young", "Mid", "Senior"])

print(df["AgeGroup"].value_counts())

plt.figure(figsize=(8, 4))
sns.countplot(x="AgeGroup", hue="LeaveOrNot", data=df)
plt.title("LeaveOrNot Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="LeaveOrNot (0 = Stayed, 1 = Left)")
plt.show()



# Check class balance for target variable ('LeaveOrNot' is the target variable)
if 'LeaveOrNot' in df.columns:
    print("\nTarget Variable Distribution:")
    print(df['LeaveOrNot'].value_counts(normalize=True))
    sns.countplot(x='LeaveOrNot', data=df)
    plt.title("Distribution of Employees Leaving")
    plt.show()
    
    
#Check outliers

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outlier_rows = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    return not outlier_rows.empty

low, up = outlier_thresholds(df, "Age")
print("Outlier Boundries for Age:", low, up)

print(df[(df["Age"] < low) | (df["Age"] > up)].head())

print(df[(df["Age"] < low) | (df["Age"] > up)].index)

print("Is there an outlier value in the column of Age?", check_outlier(df, "Age"))



# Correlation heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()


"""
According to Heatmap, JoiningYear has the highest positive correlation with turnover 
and PaymentTier has the lowest negative correlation. 
Let's take a closer look at these relationships

"""

# Relation between Joining Year and Leave or Not
plt.figure(figsize=(10, 4))
sns.histplot(data=df, x="JoiningYear", hue="LeaveOrNot", multiple="stack", kde=True, bins=15)
plt.title("Joining Year vs. LeaveOrNot")
plt.xlabel("Joining Year")
plt.ylabel("Count")
plt.show()


# Relation between Payment Tier and Leave or Not
plt.figure(figsize=(8, 4))
sns.countplot(x="PaymentTier", hue="LeaveOrNot", data=df)
plt.title("Payment Tier vs. LeaveOrNot")
plt.xlabel("Payment Tier")
plt.ylabel("Count")
plt.legend(title="LeaveOrNot (0 = Stayed, 1 = Left)")
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
#for col in numerical_features:
    #plt.figure(figsize=(8, 4))
    #sns.boxplot(x=df[col])
    #plt.title(f"Box Plot of {col}")
    #plt.show()

# Relationship between features and target variable (for classification tasks)
if 'LeaveOrNot' in df.columns:
    for col in numerical_features:
        if col != 'LeaveOrNot':
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='LeaveOrNot', y=col, data=df)
            plt.title(f"{col} vs. LeaveOrNot")
            plt.show()
            
            
# Indicates whether a certain categorical variable has an impact on employee turnover
for col in categorical_features:
    print(f"\n### {col} vs LeaveOrNot ###")
    print(pd.crosstab(df[col], df["LeaveOrNot"], normalize="index"))
    sns.countplot(x=col, hue="LeaveOrNot", data=df)
    plt.title(f"LeaveOrNot Distribution by {col}")
    plt.show()



# To analyse the skewness measure (We use the log transformation to approximate 
#the distribution of a variable to the normal distribution)

print("\nSkewness of Numerical Features:")
print(df[numerical_features].skew())

# If the skewness is greater than 1, the log transformation can be applied
for col in numerical_features:
    if abs(df[col].skew()) > 1:
        plt.figure(figsize=(8, 4))
        sns.histplot(np.log1p(df[col]), kde=True, bins=30)
        plt.title(f"Log Transformed Distribution of {col}")
        plt.show()

"""

Skewness of Numerical Features:
JoiningYear                 -0.113462 
PaymentTier                 -1.709531 (High skewnees but categorical)
Age                          0.905195
ExperienceInCurrentDomain   -0.162556
LeaveOrNot                   0.657631
dtype: float64

"""

# Apply log transformation to Age variable only
if df["Age"].min() <= 0:
    df["Age"] = df["Age"] - df["Age"].min() + 1  # If there is a zero in the values, we make it positive.

df["Age_log"] = np.log1p(df["Age"])

# Visualize new distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["Age_log"], kde=True, bins=30)
plt.title("Log Transformed Distribution of Age")
plt.show()



#----------------------------------------------------------------------------#

# MLP (Multi Layer Perceptron) Analysis

# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier

# For handling imbalanced data, we import SMOTE and Pipeline from imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ------------------------------------------------------------------
# 1. Data Preparation
# ------------------------------------------------------------------

# If your dataset has not been loaded yet, uncomment the lines below:
# file_path = r"C:\Users\deniz\Desktop\Explanaible AI\XAI_Codes\Employee.csv"
# df = pd.read_csv(file_path)

# If the "Age_log" column has not been created, generate it using log transformation:
if "Age_log" not in df.columns:
    if df["Age"].min() <= 0:
        df["Age"] = df["Age"] - df["Age"].min() + 1
    df["Age_log"] = np.log1p(df["Age"])

# Specify the features and the target column:
feature_cols = ["Education", "JoiningYear", "City", "PaymentTier", "Age_log", "Gender", "EverBenched", "ExperienceInCurrentDomain"]
target_col = "LeaveOrNot"

# Separate features (X) and the target (y):
X = df[feature_cols]
y = df[target_col]

# If there are missing values, clean them:
X = X.dropna()
y = y[X.index]

# Define numeric and categorical variables:
numeric_features = ["JoiningYear", "PaymentTier", "Age_log", "ExperienceInCurrentDomain"]
categorical_features = ["Education", "City", "Gender", "EverBenched"]

# ------------------------------------------------------------------
# 2. Creating the Pipeline
# ------------------------------------------------------------------

# For numeric features, we use StandardScaler;
# for categorical features, we use OneHotEncoder with dense output (sparse_output=False):
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# MLP Classifier - with the desired parameters:
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    learning_rate_init=0.001  # The default is usually 0.001; we have set it to this value.
)

# Create a SMOTE object:
smote = SMOTE(random_state=42)

# Create an imblearn Pipeline:
# Important: First, we add the 'preprocessor', then 'smote', and finally the 'classifier'.
imb_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', mlp_classifier)
])

# ------------------------------------------------------------------
# 3. Model Evaluation with Stratified K-Fold Cross-Validation
# ------------------------------------------------------------------

# Use StratifiedKFold with 5 folds (to maintain class distribution in each fold):
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model using cross_validate with various metrics:
cv_results = cross_validate(
    imb_pipeline, 
    X, 
    y, 
    cv=skf, 
    scoring=['accuracy', 'precision', 'recall', 'f1'], 
    return_train_score=True
)

# Print the average results:
print("Cross-Validation Results with SMOTE:")
print("Training Set (Train) Accuracy:", np.mean(cv_results['train_accuracy']))
print("Validation Set (Test) Accuracy:", np.mean(cv_results['test_accuracy']))
print("Validation Precision:", np.mean(cv_results['test_precision']))
print("Validation Recall:", np.mean(cv_results['test_recall']))
print("Validation F1-Score:", np.mean(cv_results['test_f1']))




