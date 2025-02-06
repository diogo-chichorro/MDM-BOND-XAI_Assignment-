import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from alibi.explainers import AnchorTabular

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

# Medium-Sized Neural Network Analysis with Anchors
if 'LeaveOrNot' in df.columns:
    # Prepare the data: Separate features and target variable
    target = 'LeaveOrNot'
    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical columns to numerical using label encoding
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Standardize numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a medium-sized neural network model
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.2f}")

    # Explain model predictions using Anchors
    explainer = AnchorTabular(predict_fn=lambda x: (model.predict(x) > 0.5).astype(int).flatten(), feature_names=X.columns.tolist())
    
    # Fit the explainer to the training data
    explainer.fit(X_train.values)

    # Choose a sample from the test set to explain
    sample_index = 0  # Change this index to explore different examples
    sample = X_test.iloc[sample_index].values.reshape(1, -1)

    # Generate the explanation
    explanation = explainer.explain(sample)

    # Display the explanation
    print("\nAnchor Explanation for Sample:")
    print(explanation.anchor)
    print("Precision:", explanation.precision)
    print("Coverage:", explanation.coverage)
