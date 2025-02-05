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