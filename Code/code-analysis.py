# CUSTOMER CHURN RISK PROFILER
# File: code-analysis.py

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------
# 1. FILE SETUP
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "synthetic_telecom_customers.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------
# 2. LOAD DATA
# -----------------------------------------

def load_data():
    print("\nLoading customer dataset...")

    try:
        df = pd.read_csv(DATA_PATH)
        print("Dataset loaded successfully!")
        return df

    except FileNotFoundError:
        print("ERROR: Dataset file not found.")
        print("Expected location:", DATA_PATH)
        return None


# -----------------------------------------
# 3. DATA INSPECTION
# -----------------------------------------

def inspect_data(df):
    print("\nDATASET OVERVIEW")
    print("-------------------------")

    print("Number of customers:", len(df))
    print("Number of columns:", len(df.columns))
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate customer IDs:", df["customer_id"].duplicated().sum())

    print("\nChurn distribution:")
    print(df["churn"].value_counts())

    print("\nActual churn rate:", round(df["churn"].mean() * 100, 2), "%")


# -----------------------------------------
# 4. BUSINESS KPIs
# -----------------------------------------

def calculate_kpis(df):
    print("\nBUSINESS PERFORMANCE INDICATORS")
    print("--------------------------------")

    total_customers = len(df)
    churned_customers = int(df["churn"].sum())
    retained_customers = total_customers - churned_customers

    churn_rate = df["churn"].mean() * 100
    average_monthly_charges = df["monthly_charges"].mean()
    average_tenure = df["tenure_months"].mean()

    # Sum of monthly charges for customers marked as churned
    churned_monthly_charges = df.loc[
        df["churn"] == 1, "monthly_charges"
    ].sum()

    print("Total customers:", total_customers)
    print("Churned customers:", churned_customers)
    print("Retained customers:", retained_customers)
    print("Churn rate:", round(churn_rate, 2), "%")
    print("Average monthly charges: ₹", round(average_monthly_charges, 2))
    print("Average customer tenure:", round(average_tenure, 2), "months")
    print(
        "Aggregate monthly charges of churned customers: ₹",
        round(churned_monthly_charges, 2)
    )


# -----------------------------------------
# 5. CUSTOMER SEGMENTATION
# -----------------------------------------

def segment_analysis(df):
    print("\nCUSTOMER SEGMENTATION ANALYSIS")
    print("--------------------------------")

    # Group customers by tenure without changing the original dataset
    analysis_df = df.copy()
    analysis_df["tenure_group"] = pd.cut(
        analysis_df["tenure_months"],
        bins=[0, 12, 24, 48, 72],
        labels=["1-12 months", "13-24 months", "25-48 months", "49-72 months"]
    )

    segments = [
        ("1. CHURN BY CONTRACT TYPE", "contract_type"),
        ("2. CHURN BY TENURE GROUP", "tenure_group"),
        ("3. CHURN BY INTERNET SERVICE", "internet_service"),
        ("4. CHURN BY SATISFACTION SCORE", "satisfaction_score"),
        ("5. CHURN BY SUPPORT TICKETS", "support_tickets"),
        ("6. CHURN BY PAYMENT METHOD", "payment_method"),
        ("7. CHURN BY AUTOPAY", "autopay"),
        ("8. CHURN BY TECH SUPPORT PLAN", "tech_support_plan")
    ]

    for heading, column in segments:
        print(f"\n{heading}")
        churn_rates = analysis_df.groupby(
            column, observed=True
        )["churn"].mean() * 100
        print(churn_rates.round(2))

    print("\nSUPPORT TICKET CUSTOMER COUNTS")
    support_counts = analysis_df.groupby("support_tickets").agg(
        total_customers=("churn", "count"),
        churned_customers=("churn", "sum")
    )
    print(support_counts)


# -----------------------------------------
# 6. LOGISTIC REGRESSION MODEL
# -----------------------------------------

def train_churn_model(df):
    print("\nLOGISTIC REGRESSION MODEL")
    print("--------------------------------")

    numerical_features = [
        "age",
        "tenure_months",
        "monthly_charges",
        "support_tickets",
        "satisfaction_score",
        "payment_delay_days",
        "usage_hours"
    ]

    categorical_features = [
        "city_tier",
        "contract_type",
        "internet_service",
        "autopay",
        "payment_method",
        "tech_support_plan"
    ]

    # Separate model inputs from the churn outcome
    feature_columns = numerical_features + categorical_features
    X = df[feature_columns]
    y = df["churn"]

    # Keep the final test data separate from model selection
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Split development data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev
    )

    print("\nDATA SPLIT")
    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))

    # Scale numerical inputs and encode categorical inputs
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numerical_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    # Compare standard and class-weighted Logistic Regression
    model_options = {
        "Baseline": None,
        "Balanced": "balanced"
    }

    thresholds = [0.50, 0.40, 0.30, 0.25, 0.20]

    for model_name, class_weight in model_options.items():
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight
                )
            )
        ])

        model.fit(X_train, y_train)
        validation_probabilities = model.predict_proba(X_val)[:, 1]

        print(f"\n{model_name.upper()} MODEL")
        print("-------------------------")

        # Compare classification performance at different thresholds
        for threshold in thresholds:
            validation_predictions = (
                validation_probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_val, validation_predictions, zero_division=0
            )
            recall = recall_score(
                y_val, validation_predictions, zero_division=0
            )
            f1 = f1_score(
                y_val, validation_predictions, zero_division=0
            )

            print("\nThreshold:", threshold)
            print("Precision:", round(precision, 4))
            print("Recall:", round(recall, 4))
            print("F1 Score:", round(f1, 4))

        validation_auc = roc_auc_score(y_val, validation_probabilities)
        print("\nValidation ROC-AUC:", round(validation_auc, 4))

    # Train the selected baseline model on all development data
    selected_threshold = 0.25

    final_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    final_model.fit(X_dev, y_dev)

    # Evaluate once on the untouched test data
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    print("\nFINAL MODEL EVALUATION")
    print("-------------------------")
    print("Selected model: Baseline Logistic Regression")
    print("Selected threshold:", selected_threshold)

    print("\nFINAL TEST RESULTS")
    print("Accuracy:", round(accuracy_score(y_test, test_predictions), 4))
    print(
        "Precision:",
        round(precision_score(y_test, test_predictions, zero_division=0), 4)
    )
    print(
        "Recall:",
        round(recall_score(y_test, test_predictions, zero_division=0), 4)
    )
    print("F1 Score:", round(f1_score(y_test, test_predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, test_probabilities), 4))

    print("\nFINAL CONFUSION MATRIX")
    print(confusion_matrix(y_test, test_predictions))
    print("\nFINAL MODEL EVALUATION COMPLETED")

    return final_model, X_test, y_test


# -----------------------------------------
# 7. CUSTOMER RISK PROFILING
# -----------------------------------------

def generate_customer_risk_profiles(df, final_model):
    print("\nCUSTOMER-LEVEL RISK PROFILING")
    print("------------------------------")

    # Exclude identifiers, the target and derived total charges from prediction
    X_customers = df.drop(
        columns=["customer_id", "total_charges", "churn"]
    )

    risk_df = df.copy()
    risk_df["churn_probability"] = final_model.predict_proba(
        X_customers
    )[:, 1]

    # Convert predicted probabilities into three risk groups
    risk_df["risk_category"] = pd.cut(
        risk_df["churn_probability"],
        bins=[0, 0.15, 0.30, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    risk_distribution = risk_df["risk_category"].value_counts().sort_index()

    print("\nCUSTOMER RISK CATEGORY DISTRIBUTION")
    print(risk_distribution)

    print("\nRISK CATEGORY PERCENTAGES")
    print((risk_distribution / len(risk_df) * 100).round(2))

    # Compare observed churn rates across the predicted risk groups
    print("\nACTUAL CHURN RATE BY RISK CATEGORY")
    churn_by_risk = risk_df.groupby(
        "risk_category", observed=False
    )["churn"].mean() * 100
    print(churn_by_risk.round(2))

    print("\nCHURN PROBABILITY DISTRIBUTION")
    print(risk_df["churn_probability"].describe().round(3))

    print("\nTOP 10 CUSTOMERS BY PREDICTED CHURN RISK")
    top_customers = risk_df.nlargest(10, "churn_probability")

    print(
        top_customers[
            [
                "customer_id",
                "tenure_months",
                "contract_type",
                "churn_probability"
            ]
        ].to_string(index=False)
    )

    return risk_df


# -----------------------------------------
# 8. CHURN DRIVER ANALYSIS
# -----------------------------------------

def analyze_churn_drivers(final_model, X_test, y_test):
    print("\nTOP CHURN-ASSOCIATED FACTORS")
    print("--------------------------------")

    # Measure the drop in ROC-AUC when each feature is shuffled
    importance = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": importance.importances_mean,
        "std_deviation": importance.importances_std
    }).sort_values(
        by="importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE RANKING")
    print(importance_df.round(4).to_string(index=False))

    print("\nTOP THREE CHURN-ASSOCIATED FACTORS")
    for rank, row in enumerate(
        importance_df.head(3).itertuples(index=False), start=1
    ):
        print(
            f"{rank}. {row.feature} "
            f"(Importance: {row.importance:.4f})"
        )

    return importance_df


# -----------------------------------------
# 9. MAIN PROGRAM
# -----------------------------------------

def main():
    print("CUSTOMER CHURN ANALYSIS")
    print("========================")

    df = load_data()

    if df is None:
        return

    inspect_data(df)
    calculate_kpis(df)
    segment_analysis(df)

    final_model, X_test, y_test = train_churn_model(df)
    risk_df = generate_customer_risk_profiles(df, final_model)
    importance_df = analyze_churn_drivers(final_model, X_test, y_test)

    # Save outputs required by the visualization script
    risk_df.to_csv(OUTPUT_DIR / "customer_risk_profiles.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print("\nAnalysis results saved successfully!")
    print("\nData analysis and model training completed.")


if __name__ == "__main__":
    main()
