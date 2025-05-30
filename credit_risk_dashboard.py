# Import required libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Configure Streamlit page
st.set_page_config(
    page_title="Credit Risk Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("📊 Credit Risk Analysis Dashboard")
st.markdown(
    """
    Welcome to the **Credit Risk Analysis Dashboard**. This Dashboard provides insights into loan data,
    explores key trends, evaluates a machine learning model, and visualizes key features.
    """
)

# Load Dataset
st.subheader("📂 Dataset Overview")
try:
    # Load a sample of the Lending Club Loan Dataset
    data = pd.read_csv('LendingClubLoanData.csv.gz', nrows=1000, compression='gzip')
    st.success("Dataset Loaded Successfully!")
    st.write("### First 5 Rows of the Dataset")
    st.write(data.head())
    st.write("### Dataset Shape:", data.shape)

    # Check for missing values
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    st.write("### Missing Values (in %):")
    st.write(missing_percentage)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Data Cleaning
data = data.dropna(thresh=len(data) * 0.7, axis=1)  # Drop columns with more than 30% missing values
data = data.drop(['id', 'member_id'], axis=1, errors='ignore')  # Drop irrelevant columns

# Fill missing values
data_numeric = data.select_dtypes(include=['number'])
data[data_numeric.columns] = data_numeric.fillna(data_numeric.mean())
data_non_numeric = data.select_dtypes(include=['object'])
for col in data_non_numeric.columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Drop rows with missing target variable
data = data.dropna(subset=['loan_status']).reset_index(drop=True)

st.write("### Cleaned Dataset Shape:", data.shape)

# Exploratory Data Analysis (EDA)
st.subheader("📊 Exploratory Data Analysis")

# Loan Status Distribution
st.write("### Loan Status Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.countplot(x='loan_status', data=data, ax=ax1)
ax1.set_title('Loan Status Distribution')
ax1.set_xlabel('Loan Status')
ax1.set_ylabel('Count')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
st.pyplot(fig1)

# Interest Rate Distribution
st.write("### Interest Rate Distribution")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.histplot(data['int_rate'], kde=True, color='orange', ax=ax2)
ax2.set_title('Interest Rate Distribution')
ax2.set_xlabel('Interest Rate')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)

# Interest Rate by Loan Grade
st.write("### Interest Rate by Loan Grade")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(x='grade', y='int_rate', data=data, ax=ax3)
ax3.set_title('Interest Rate by Loan Grade')
ax3.set_xlabel('Loan Grade')
ax3.set_ylabel('Interest Rate')
st.pyplot(fig3)

# Model Training and Evaluation
st.subheader("🤖 Model Training and Evaluation")

# Define Features (X) and Target (y)
X = data.drop('loan_status', axis=1, errors='ignore')  # Replace 'loan_status' with your target column
y = data['loan_status']

# Dynamically Identify Categorical Columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Apply One-Hot Encoding to Categorical Columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Align Train and Test Sets
X_train, X_test = X_train.align(X_test, join='left', axis=1)
X_test = X_test.fillna(0)  # Fill missing columns in X_test with 0

# Convert Target Variable to Numeric
y = pd.get_dummies(y, drop_first=True)

# Train the Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display Model Accuracy and Classification Report
st.write("### Model Accuracy")
st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%")

st.write("### Classification Report")
st.text(classification_rep)

# Footer
st.markdown("---")
st.markdown("**Created by Harismitha | 2025**")