"""
Customer Churn Risk Profiler — Synthetic Data Generator
-------------------------------------------------------
Generates an independently simulated telecom customer dataset.

Run from the project root:
    python Code/synthetic-data-gen.py

Output:
    Code/data/synthetic_telecom_customers.csv

The IBM Telco Customer Churn dataset is used only as a design reference.
No IBM customer records or outcomes are copied.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# 1. Project settings
# -----------------------------

RANDOM_SEED = 42
NUMBER_OF_CUSTOMERS = 13_500
TARGET_EXPECTED_CHURN_RATE = 0.20

SCRIPT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = SCRIPT_FOLDER / "data"
OUTPUT_FILE = DATA_FOLDER / "synthetic_telecom_customers.csv"


# -----------------------------
# 2. Helper functions
# -----------------------------

def sigmoid(values):
    """Convert risk scores into probabilities between 0 and 1."""
    # Clipping prevents numerical overflow for unusually large scores.
    values = np.clip(values, -30, 30)
    return 1 / (1 + np.exp(-values))


def calibrate_intercept(risk_without_intercept, target_rate):
    """
    Find an intercept that makes the average churn probability close
    to the requested design target.

    The final binary churn rate will vary slightly because outcomes
    are randomly drawn from the calculated probabilities.
    """
    lower_bound = -10
    upper_bound = 10

    for _ in range(100):
        midpoint = (lower_bound + upper_bound) / 2
        average_probability = sigmoid(
            risk_without_intercept + midpoint
        ).mean()

        if average_probability < target_rate:
            lower_bound = midpoint
        else:
            upper_bound = midpoint

    return (lower_bound + upper_bound) / 2


# -----------------------------
# 3. Generate customer attributes
# -----------------------------

def generate_customer_data(number_of_customers=NUMBER_OF_CUSTOMERS,
                           random_seed=RANDOM_SEED):
    """Generate the complete synthetic telecom customer dataset."""

    rng = np.random.default_rng(random_seed)
    number = number_of_customers

    # Unique customer identifiers
    customer_id = [
        f"CUST{number_id:05d}"
        for number_id in range(1, number + 1)
    ]

    # Customer demographics
    age = np.rint(
        rng.normal(loc=38, scale=12, size=number)
    ).clip(18, 75).astype(int)

    city_tier = rng.choice(
        ["Metro", "Tier-2", "Tier-3"],
        size=number,
        p=[0.35, 0.40, 0.25]
    )

    # Customer tenure
    tenure_months = rng.integers(1, 73, size=number)

    # Longer-tenure customers have a greater chance of longer contracts.
    contract_type = []
    for tenure in tenure_months:
        if tenure <= 12:
            probabilities = [0.75, 0.17, 0.08]
        elif tenure <= 36:
            probabilities = [0.55, 0.27, 0.18]
        else:
            probabilities = [0.38, 0.32, 0.30]

        contract_type.append(
            rng.choice(
                ["Month-to-month", "One year", "Two year"],
                p=probabilities
            )
        )

    contract_type = np.array(contract_type)

    # Service type
    internet_service = rng.choice(
        ["Mobile only", "Broadband", "Fiber"],
        size=number,
        p=[0.30, 0.30, 0.40]
    )

    # Monthly charges vary by service type, with random customer variation.
    service_base = {
        "Mobile only": 350,
        "Broadband": 750,
        "Fiber": 1_050
    }

    base_charges = np.array(
        [service_base[service] for service in internet_service]
    )

    monthly_charges = (
        base_charges + rng.normal(loc=0, scale=150, size=number)
    )
    monthly_charges = np.clip(monthly_charges, 200, 1_500)
    monthly_charges = np.round(monthly_charges, 2)

    # Support tickets: count data, with a maximum of 8.
    support_tickets = np.clip(
        rng.poisson(lam=1.5, size=number),
        0,
        8
    ).astype(int)

    # Satisfaction is influenced by support experience, but remains noisy.
    satisfaction_score = np.rint(
        4.2 - (support_tickets * 0.32)
        + rng.normal(loc=0, scale=0.85, size=number)
    )
    satisfaction_score = np.clip(
        satisfaction_score, 1, 5
    ).astype(int)

    # Autopay status
    autopay = rng.choice(
        ["Yes", "No"],
        size=number,
        p=[0.55, 0.45]
    )

    # Payment method is loosely associated with autopay status.
    payment_method = []
    for uses_autopay in autopay:
        if uses_autopay == "Yes":
            probabilities = [0.30, 0.20, 0.12, 0.35, 0.03]
        else:
            probabilities = [0.25, 0.18, 0.22, 0.05, 0.30]

        payment_method.append(
            rng.choice(
                ["UPI", "Credit Card", "Debit Card",
                 "Bank Auto-debit", "Cash/Wallet"],
                p=probabilities
            )
        )

    payment_method = np.array(payment_method)

    # Payment delays: autopay customers generally have fewer delays.
    payment_delay_days = np.zeros(number, dtype=int)

    autopay_mask = autopay == "Yes"
    non_autopay_mask = ~autopay_mask

    payment_delay_days[autopay_mask] = np.where(
        rng.random(autopay_mask.sum()) < 0.88,
        0,
        rng.integers(1, 8, size=autopay_mask.sum())
    )

    payment_delay_days[non_autopay_mask] = np.where(
        rng.random(non_autopay_mask.sum()) < 0.42,
        0,
        rng.integers(1, 31, size=non_autopay_mask.sum())
    )

    # Usage hours
    usage_hours = rng.normal(
        loc=120, scale=40, size=number
    )
    usage_hours = np.clip(usage_hours, 10, 300)
    usage_hours = np.round(usage_hours, 2)

    # Tech support plan: availability/uptake varies by service type.
    tech_support_plan = []
    for service in internet_service:
        if service == "Mobile only":
            probability_yes = 0.20
        elif service == "Broadband":
            probability_yes = 0.45
        else:
            probability_yes = 0.60

        tech_support_plan.append(
            "Yes" if rng.random() < probability_yes else "No"
        )

    tech_support_plan = np.array(tech_support_plan)

    # Approximate accumulated charges.
    total_charges = np.round(
        tenure_months * monthly_charges, 2
    )

    # -----------------------------
    # 4. Generate churn outcomes
    # -----------------------------

    # Each factor is centred around a reference value. This makes the
    # intercept calibration easier to interpret.
    tenure_risk = (36 - tenure_months) / 18

    contract_risk = np.select(
        [
            contract_type == "Month-to-month",
            contract_type == "One year",
            contract_type == "Two year"
        ],
        [0.75, -0.15, -0.55],
        default=0
    )

    charge_risk = (monthly_charges - 750) / 400
    support_risk = (support_tickets - 1.5) / 2
    satisfaction_risk = (3 - satisfaction_score) / 1.5
    payment_delay_risk = payment_delay_days / 15
    autopay_risk = np.where(autopay == "No", 0.25, 0)
    tech_support_risk = np.where(
        tech_support_plan == "No", 0.15, 0
    )
    usage_risk = (usage_hours - 120) / 100

    # Weighted risk score. Age and city tier are deliberately not
    # direct contributors to churn risk in this simulation.
    risk_without_intercept = (
        0.65 * tenure_risk
        + 0.70 * contract_risk
        + 0.25 * charge_risk
        + 0.45 * support_risk
        + 0.65 * satisfaction_risk
        + 0.35 * payment_delay_risk
        + 0.15 * autopay_risk
        + 0.12 * tech_support_risk
        + 0.08 * usage_risk
    )

    intercept = calibrate_intercept(
        risk_without_intercept,
        TARGET_EXPECTED_CHURN_RATE
    )

    churn_probability = sigmoid(
        risk_without_intercept + intercept
    )

    # Random outcome: 1 = churned, 0 = retained.
    churn = (
        rng.random(number) < churn_probability
    ).astype(int)

    # -----------------------------
    # 5. Assemble the dataset
    # -----------------------------

    customer_data = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "city_tier": city_tier,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "monthly_charges": monthly_charges,
        "support_tickets": support_tickets,
        "satisfaction_score": satisfaction_score,
        "autopay": autopay,
        "payment_delay_days": payment_delay_days,
        "usage_hours": usage_hours,
        "payment_method": payment_method,
        "tech_support_plan": tech_support_plan,
        "total_charges": total_charges,
        "churn": churn
    })

    return customer_data


# -----------------------------
# 6. Validate and save the data
# -----------------------------

def validate_dataset(data):
    """Check the dataset's basic quality and expected structure."""

    required_columns = [
        "customer_id", "age", "city_tier", "tenure_months",
        "contract_type", "internet_service", "monthly_charges",
        "support_tickets", "satisfaction_score", "autopay",
        "payment_delay_days", "usage_hours", "payment_method",
        "tech_support_plan", "total_charges", "churn"
    ]

    if data.empty:
        raise ValueError("The generated dataset is empty.")

    if list(data.columns) != required_columns:
        raise ValueError("The dataset columns do not match the specification.")

    if data["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs were found.")

    if data.isnull().any().any():
        raise ValueError("Missing values were found.")

    if not data["age"].between(18, 75).all():
        raise ValueError("Age values are outside the permitted range.")

    if not data["tenure_months"].between(1, 72).all():
        raise ValueError("Tenure values are outside the permitted range.")

    if not data["monthly_charges"].between(200, 1500).all():
        raise ValueError("Monthly charges are outside the permitted range.")

    if not data["support_tickets"].between(0, 8).all():
        raise ValueError("Support ticket values are outside the permitted range.")

    if not data["satisfaction_score"].between(1, 5).all():
        raise ValueError("Satisfaction scores are outside the permitted range.")

    if not data["payment_delay_days"].between(0, 30).all():
        raise ValueError("Payment delay values are outside the permitted range.")

    if not data["usage_hours"].between(10, 300).all():
        raise ValueError("Usage hours are outside the permitted range.")

    if not data["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 and 1.")

    return True


def save_dataset(data, output_file=OUTPUT_FILE):
    """Create the data folder if needed and save the dataset as CSV."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)
    return output_file


# -----------------------------
# 7. Main program
# -----------------------------

def main():
    """Run the complete synthetic data generation process."""

    print("Generating synthetic telecom customer data...")

    customer_data = generate_customer_data()
    validate_dataset(customer_data)
    saved_file = save_dataset(customer_data)

    churn_rate = customer_data["churn"].mean() * 100

    print("\nDATA GENERATION SUCCESSFUL")
    print("-------------------------")
    print(f"Number of customers: {len(customer_data):,}")
    print(f"Number of columns: {len(customer_data.columns)}")
    print(f"Churned customers: {customer_data['churn'].sum():,}")
    print(f"Retained customers: {(customer_data['churn'] == 0).sum():,}")
    print(f"Actual churn rate: {churn_rate:.2f}%")
    print(f"Missing values: {customer_data.isnull().sum().sum()}")
    print(f"Duplicate customer IDs: {customer_data['customer_id'].duplicated().sum()}")
    print(f"\nDataset saved to: {saved_file}")

    print("\nFirst five records:")
    print(customer_data.head().to_string(index=False))

    print("\nDataset information:")
    customer_data.info()

    print("\nDescriptive statistics:")
    print(customer_data.describe().round(2))


if __name__ == "__main__":
    main()
