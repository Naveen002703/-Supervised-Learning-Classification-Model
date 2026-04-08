import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Classification Model", layout="wide")

st.title("🧠 Lab Program 6: Supervised Learning – Classification Model")
st.subheader("Logistic Regression & Decision Tree")

# ------------------------------------------------
# Upload Dataset
# ------------------------------------------------
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📌 Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)

    # ------------------------------------------------
    # Select Target
    # ------------------------------------------------
    st.subheader("🎯 Select Target Variable")
    target = st.selectbox("Choose Target Column", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Use only numeric features
        X = X.select_dtypes(include=np.number)

        st.write("Feature Columns Used:", X.columns.tolist())

        # ------------------------------------------------
        # Model Selection
        # ------------------------------------------------
        model_choice = st.selectbox(
            "Choose Classification Model",
            ["Logistic Regression", "Decision Tree"]
        )

        # ------------------------------------------------
        # Train-Test Split
        # ------------------------------------------------
        test_size = st.slider("Select Test Size (%)", 10, 40, 20) / 100

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        st.success(f"Train Size: {X_train.shape[0]} rows")
        st.success(f"Test Size: {X_test.shape[0]} rows")

        # ------------------------------------------------
        # Train Model
        # ------------------------------------------------
        if st.button("Train Model"):

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = DecisionTreeClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # ------------------------------------------------
            # Evaluation Metrics
            # ------------------------------------------------
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            st.subheader("📊 Model Evaluation")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", round(accuracy, 4))
            col2.metric("Precision", round(precision, 4))
            col3.metric("Recall", round(recall, 4))
            col4.metric("F1-Score", round(f1, 4))

            # ------------------------------------------------
            # Confusion Matrix
            # ------------------------------------------------
            st.subheader("🔎 Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            cax = ax.matshow(cm)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.colorbar(cax)

            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, val, ha='center', va='center')

            st.pyplot(fig)

else:
    st.info("Please upload a CSV dataset to begin.")

# ------------------------------------------------
# Theory Section
# ------------------------------------------------
st.markdown("---")
st.subheader("📘 Theory Explanation")

st.markdown("""
### 🔹 Logistic Regression
Used for binary classification problems.

Uses sigmoid function:
\[
P(Y=1) = \frac{1}{1 + e^{-z}}
\]

---

### 🔹 Decision Tree
- Tree-based model
- Splits data using feature conditions
- Easy to interpret

---

### 🔹 Evaluation Metrics

1️⃣ Accuracy  
\[
Accuracy = \frac{TP + TN}{Total}
\]

2️⃣ Precision  
\[
Precision = \frac{TP}{TP + FP}
\]

3️⃣ Recall  
\[
Recall = \frac{TP}{TP + FN}
\]

4️⃣ F1-Score  
\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

---

### 🔹 Confusion Matrix

|            | Predicted Positive | Predicted Negative |
|------------|-------------------|-------------------|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

""")