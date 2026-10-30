
# CUSTOMER CHURN RISK PROFILER
# File: code-visualization.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------
# 1. FILE SETUP AND CHART STYLE
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "synthetic_telecom_customers.csv"
RISK_PATH = BASE_DIR / "output" / "customer_risk_profiles.csv"
IMPORTANCE_PATH = BASE_DIR / "output" / "feature_importance.csv"

CHART_DIR = BASE_DIR / "output" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Corporate colour palette
BLUE = "#2471A3"
GREEN = "#239B56"
RED = "#E74C3C"
ORANGE = "#F39C12"

# Apply consistent formatting to all charts
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.figsize": (9, 5.5)
})


# -----------------------------------------
# 2. LOAD AND VALIDATE INPUT FILES
# -----------------------------------------

def load_visualization_data():
    """Load and validate the input files required for visualisation."""

    required_files = [DATA_PATH, RISK_PATH, IMPORTANCE_PATH]

    # Check that all required files exist
    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {file_path}\n"
                "Run code-analysis.py first."
            )

    # Read the CSV files
    try:
        df = pd.read_csv(DATA_PATH)
        risk_df = pd.read_csv(RISK_PATH)
        importance_df = pd.read_csv(IMPORTANCE_PATH)

    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError(
            f"Could not read an input CSV file: {error}"
        ) from error

    # Check for empty datasets
    if df.empty or risk_df.empty or importance_df.empty:
        raise ValueError("One or more input files contain no data.")

    # Required columns for each file
    required_data_columns = {
        "churn", "contract_type",
        "tenure_months", "satisfaction_score"
    }

    required_risk_columns = {"risk_category"}
    required_importance_columns = {"feature", "importance"}

    if not required_data_columns.issubset(df.columns):
        raise ValueError("Customer dataset is missing required columns.")

    if not required_risk_columns.issubset(risk_df.columns):
        raise ValueError("Risk profile file is missing risk_category.")

    if not required_importance_columns.issubset(importance_df.columns):
        raise ValueError("Feature importance file is missing required columns.")

    # Check numerical columns
    numerical_columns = [
        (df, "churn"),
        (df, "tenure_months"),
        (df, "satisfaction_score"),
        (importance_df, "importance")
    ]

    for data, column in numerical_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{column} must contain numeric values.")

        if not np.isfinite(data[column]).all():
            raise ValueError(f"{column} contains missing or invalid numbers.")

    return df, risk_df, importance_df


# -----------------------------------------
# 3. CHART 1: CUSTOMER CHURN DISTRIBUTION
# -----------------------------------------

def plot_churn_distribution(df):
    """Show retained and churned customer counts and percentages."""

    churn_counts = df["churn"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    total_customers = len(df)
    percentages = churn_counts / total_customers * 100

    labels = ["Retained", "Churned"]
    colours = [GREEN, RED]

    fig, ax = plt.subplots()

    bars = ax.bar(
        labels,
        churn_counts.values,
        color=colours,
        width=0.6
    )

    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Number of Customers")

    # Display both customer counts and percentages
    bar_labels = [
        f"{count:,}\n({percentage:.2f}%)"
        for count, percentage in zip(
            churn_counts.values, percentages.values
        )
    ]

    ax.bar_label(bars, labels=bar_labels, padding=4)
    ax.set_ylim(0, max(churn_counts.max() * 1.18, 1))

    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "01_churn_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------------------
# 4. CHART 2: CHURN BY CONTRACT TYPE
# -----------------------------------------

def plot_churn_by_contract(df):
    """Compare churn rates across contract types."""

    # Mean of a binary churn column gives the proportion churned
    churn_rates = (
        df.groupby("contract_type", observed=True)["churn"].mean() * 100
    ).sort_values(ascending=False)

    fig, ax = plt.subplots()

    bars = ax.bar(
        churn_rates.index.astype(str),
        churn_rates.values,
        color=BLUE,
        width=0.6
    )

    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, max(churn_rates.max() * 1.2, 5))

    ax.bar_label(bars, fmt="%.1f%%", padding=4)

    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "02_churn_by_contract.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------------------
# 5. CHART 3: CHURN BY CUSTOMER TENURE
# -----------------------------------------

def plot_churn_by_tenure(df):
    """Compare churn rates across customer tenure groups."""

    analysis_df = df.copy()

    # Group customers into four tenure ranges
    analysis_df["tenure_group"] = pd.cut(
        analysis_df["tenure_months"],
        bins=[0, 12, 24, 48, 72],
        labels=[
            "1–12 months",
            "13–24 months",
            "25–48 months",
            "49–72 months"
        ],
        include_lowest=True
    )

    churn_rates = (
        analysis_df.groupby(
            "tenure_group", observed=True
        )["churn"].mean() * 100
    )

    fig, ax = plt.subplots()

    bars = ax.bar(
        churn_rates.index.astype(str),
        churn_rates.values,
        color=BLUE,
        width=0.6
    )

    ax.set_title("Churn Rate by Customer Tenure")
    ax.set_xlabel("Customer Tenure")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, max(churn_rates.max() * 1.2, 5))

    ax.bar_label(bars, fmt="%.1f%%", padding=4)

    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "03_churn_by_tenure.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------------------
# 6. CHART 4: CHURN BY SATISFACTION SCORE
# -----------------------------------------

def plot_churn_by_satisfaction(df):
    """Compare churn rates across satisfaction scores."""

    churn_rates = (
        df.groupby(
            "satisfaction_score", observed=True
        )["churn"].mean() * 100
    ).sort_index()

    fig, ax = plt.subplots()

    bars = ax.bar(
        churn_rates.index.astype(str),
        churn_rates.values,
        color=BLUE,
        width=0.6
    )

    ax.set_title("Churn Rate by Customer Satisfaction")
    ax.set_xlabel("Satisfaction Score (1 = Low, 5 = High)")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, max(churn_rates.max() * 1.2, 5))

    ax.bar_label(bars, fmt="%.1f%%", padding=4)

    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "04_churn_by_satisfaction.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------------------
# 7. CHART 5: CUSTOMER RISK DISTRIBUTION
# -----------------------------------------

def plot_risk_distribution(risk_df):
    """Show customer counts and percentages in each risk group."""

    risk_order = ["Low", "Medium", "High"]

    risk_counts = (
        risk_df["risk_category"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
    )

    total_customers = len(risk_df)
    percentages = risk_counts / total_customers * 100

    risk_colours = [GREEN, ORANGE, RED]

    fig, ax = plt.subplots()

    bars = ax.bar(
        risk_counts.index,
        risk_counts.values,
        color=risk_colours,
        width=0.6
    )

    ax.set_title("Customer Distribution by Predicted Churn Risk")
    ax.set_xlabel("Predicted Risk Category")
    ax.set_ylabel("Number of Customers")

    # Display both counts and percentages
    bar_labels = [
        f"{count:,}\n({percentage:.2f}%)"
        for count, percentage in zip(
            risk_counts.values, percentages.values
        )
    ]

    ax.bar_label(bars, labels=bar_labels, padding=4)
    ax.set_ylim(0, max(risk_counts.max() * 1.18, 1))

    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "05_customer_risk_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------------------
# 8. CHART 6: TOP CHURN-ASSOCIATED FACTORS
# -----------------------------------------

def plot_top_churn_factors(importance_df):
    """Visualise the six features with the highest permutation importance."""

    top_factors = (
        importance_df.nlargest(6, "importance")
        .sort_values("importance", ascending=True)
    )

    # Replace technical names with readable business terms
    feature_labels = {
        "tenure_months": "Customer Tenure",
        "contract_type": "Contract Type",
        "satisfaction_score": "Customer Satisfaction",
        "support_tickets": "Support Tickets",
        "monthly_charges": "Monthly Charges",
        "payment_delay_days": "Payment Delays",
        "payment_method": "Payment Method",
        "age": "Customer Age",
        "usage_hours": "Usage Hours",
        "internet_service": "Internet Service",
        "city_tier": "City Tier",
        "autopay": "Autopay",
        "tech_support_plan": "Tech Support Plan"
    }

    display_labels = [
        feature_labels.get(feature, feature)
        for feature in top_factors["feature"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.barh(
        display_labels,
        top_factors["importance"],
        color=BLUE,
        height=0.6
    )

    ax.set_title("Top Predictive Factors Associated with Churn")
    ax.set_xlabel("Permutation Importance (ROC-AUC decrease)")
    ax.set_ylabel("Customer Feature")

    maximum_importance = top_factors["importance"].max()
    ax.set_xlim(0, max(maximum_importance * 1.25, 0.01))

    ax.bar_label(bars, fmt="%.4f", padding=5)

    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "06_top_churn_factors.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------------------
# 9. CHART 7: CHURN RATE HEATMAP
# -----------------------------------------

def plot_churn_heatmap(df):
    """
    Show churn rates across combinations of contract type
    and customer tenure.
    """

    analysis_df = df.copy()

    # Create tenure groups for the heatmap
    analysis_df["tenure_group"] = pd.cut(
        analysis_df["tenure_months"],
        bins=[0, 12, 24, 48, 72],
        labels=[
            "1–12 months",
            "13–24 months",
            "25–48 months",
            "49–72 months"
        ],
        include_lowest=True
    )

    # Calculate churn rates for each contract-tenure combination
    heatmap_data = (
        analysis_df.pivot_table(
            index="contract_type",
            columns="tenure_group",
            values="churn",
            aggfunc="mean",
            observed=True
        ) * 100
    )

    contract_order = ["Month-to-month", "One year", "Two year"]
    tenure_order = [
        "1–12 months",
        "13–24 months",
        "25–48 months",
        "49–72 months"
    ]

    heatmap_data = heatmap_data.reindex(
        index=contract_order,
        columns=tenure_order
    )

    values = heatmap_data.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Redder cells indicate higher observed churn rates
    image = ax.imshow(
        masked_values,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax.set_title("Customer Churn Rate by Contract Type and Tenure")
    ax.set_xlabel("Customer Tenure")
    ax.set_ylabel("Contract Type")

    ax.set_xticks(range(len(tenure_order)))
    ax.set_xticklabels(tenure_order)

    ax.set_yticks(range(len(contract_order)))
    ax.set_yticklabels(contract_order)

    # Write the churn percentage inside each available cell
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):

            value = values[row, column]

            if np.isfinite(value):
                text_colour = "white" if value >= 55 else "black"

                ax.text(
                    column,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=text_colour,
                    fontweight="bold"
                )

    # Add a colour scale to explain the heatmap
    colourbar = fig.colorbar(image, ax=ax)
    colourbar.set_label("Churn Rate (%)")

    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "07_churn_heatmap_contract_tenure.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

# -----------------------------------------
# 10. CHART 8: CONFUSION MATRIX
# -----------------------------------------

def plot_confusion_matrix():
    """Visualise the final Logistic Regression confusion matrix."""

    values = np.array([
        [1670, 502],
        [219, 309]
    ])

    labels = [
        ["True Negative\n(TN)", "False Positive\n(FP)"],
        ["False Negative\n(FN)", "True Positive\n(TP)"]
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(values, cmap="Blues")

    ax.set_title("Logistic Regression Confusion Matrix")
    ax.set_xlabel("Predicted Outcome")
    ax.set_ylabel("Actual Outcome")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained", "Churned"])

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Retained", "Churned"])

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{labels[row][column]}\n{values[row, column]:,}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold"
            )

    fig.colorbar(image, ax=ax, label="Number of Customers")

    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "08_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------------------
# 11. MAIN PROGRAM
# -----------------------------------------

def main():
    """Load the input files and generate all seven charts."""

    print("CUSTOMER CHURN VISUALIZATION")
    print("============================")

    try:
        df, risk_df, importance_df = load_visualization_data()

        print("Input files loaded successfully.")

        # Generate each visualisation
        plot_churn_distribution(df)
        print("Created: Churn distribution")

        plot_churn_by_contract(df)
        print("Created: Churn by contract type")

        plot_churn_by_tenure(df)
        print("Created: Churn by tenure")

        plot_churn_by_satisfaction(df)
        print("Created: Churn by satisfaction")

        plot_risk_distribution(risk_df)
        print("Created: Customer risk distribution")

        plot_top_churn_factors(importance_df)
        print("Created: Top churn-associated factors")

        plot_churn_heatmap(df)
        print("Created: Contract and tenure churn heatmap")

        plot_confusion_matrix()
        print("Created: Logistic Regression confusion matrix")

        print("\nAll eight charts saved successfully.")
        print("Chart folder:", CHART_DIR)

    except (FileNotFoundError, ValueError, KeyError) as error:
        print("\nERROR:", error)


if __name__ == "__main__":
    main()
