"""
TeleGuard: Telecom Retention & Churn Intelligence
Menu-driven interface for the telecom customer retention project.

Place this file inside the Code folder, alongside:
- code-analysis.py
- code-visualization.py
- data/synthetic_telecom_customers.csv
- output/

Run:
    python menu.py
"""

from pathlib import Path
import importlib.util

import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT FILE SETUP
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_FILE = BASE_DIR / "code-analysis.py"
DATA_FILE = BASE_DIR / "data" / "synthetic_telecom_customers.csv"
OUTPUT_DIR = BASE_DIR / "output"

NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "satisfaction_score",
    "payment_delay_days",
    "usage_hours",
]

CATEGORICAL_FEATURES = [
    "city_tier",
    "contract_type",
    "internet_service",
    "autopay",
    "payment_method",
    "tech_support_plan",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

VALID_CATEGORIES = {
    "city_tier": ["Metro", "Tier-2", "Tier-3"],
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["Mobile only", "Broadband", "Fiber"],
    "autopay": ["Yes", "No"],
    "payment_method": [
        "UPI", "Credit Card", "Debit Card",
        "Bank Auto-debit", "Cash/Wallet"
    ],
    "tech_support_plan": ["Yes", "No"],
}

NUMERICAL_RANGES = {
    "age": (18, 75, "whole number"),
    "tenure_months": (1, 72, "whole number"),
    "monthly_charges": (200, 1500, "number; decimals allowed"),
    "support_tickets": (0, 8, "whole number"),
    "satisfaction_score": (1, 5, "whole number"),
    "payment_delay_days": (0, 30, "whole number"),
    "usage_hours": (10, 300, "number; decimals allowed"),
}


# ---------------------------------------------------------
# 2. GENERAL INPUT AND DISPLAY HELPERS
# ---------------------------------------------------------

def print_line():
    print("-" * 68)


def pause():
    input("\nPress Enter to continue...")


def get_menu_choice(prompt, valid_choices):
    """Read a numbered menu choice without allowing invalid input to crash."""
    while True:
        raw_value = input(prompt).strip()

        if not raw_value:
            print("Please enter a menu number.")
            continue

        try:
            choice = int(raw_value)
        except ValueError:
            print("Invalid input. Enter a whole-number menu option.")
            continue

        if choice not in valid_choices:
            print(
                "That option is not available. Choose from:",
                ", ".join(str(item) for item in valid_choices)
            )
            continue

        return choice


def get_numeric_input(field_name, minimum, maximum, number_type):
    """Collect and validate a numerical customer attribute."""
    while True:
        raw_value = input(
            f"Enter {field_name.replace('_', ' ')} "
            f"({minimum} to {maximum}): "
        ).strip()

        if not raw_value:
            print("This field cannot be blank. Please try again.")
            continue

        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if not minimum <= value <= maximum:
            print(f"Value must be between {minimum} and {maximum}.")
            continue

        if number_type == "whole number":
            if not value.is_integer():
                print("Please enter a whole number.")
                continue
            return int(value)

        return value


def get_category_input(field_name, allowed_values):
    """Collect a category, accepting case-insensitive exact matches."""
    print(f"\n{field_name.replace('_', ' ').title()}:")
    for index, category in enumerate(allowed_values, start=1):
        print(f"  {index}. {category}")

    while True:
        raw_value = input("Enter the option number or category name: ").strip()

        if not raw_value:
            print("This field cannot be blank. Please try again.")
            continue

        if raw_value.isdigit():
            category_number = int(raw_value)
            if 1 <= category_number <= len(allowed_values):
                return allowed_values[category_number - 1]
            print("That category number is not available.")
            continue

        for category in allowed_values:
            if raw_value.casefold() == category.casefold():
                return category

        print("Invalid category. Please select one of the displayed options.")


def load_analysis_module():
    """Load the existing analysis script so the menu reuses its real model."""
    if not ANALYSIS_FILE.is_file():
        raise FileNotFoundError(
            f"Could not find {ANALYSIS_FILE.name}.\n"
            "Place menu.py in the same Code folder as code-analysis.py."
        )

    spec = importlib.util.spec_from_file_location(
        "customer_churn_analysis", ANALYSIS_FILE
    )
    if spec is None or spec.loader is None:
        raise ImportError("The analysis module could not be loaded.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------
# 3. MAIN MENU AND INTRODUCTION
# ---------------------------------------------------------

def show_welcome_screen():
    print("\n" + "=" * 68)
    print("       TeleGuard")
    print("       TELECOM RETENTION & CHURN INTELLIGENCE")
    print("=" * 68)
    print(
        "\nA data-driven decision-support tool that helps a telecom "
        "retention team\nunderstand customer churn patterns, explore "
        "customer risk and prioritise\nretention activity."
    )
    print(
        "\nThe system uses an independently generated synthetic "
        "customer dataset\nand a Logistic Regression model. Its "
        "predictions are estimates, not guarantees."
    )
    print_line()
    print("MAIN MENU")
    print_line()
    print("1. Analyse Customer Churn")
    print("2. Check a Customer's Churn Risk")
    print("3. Help & Guide")
    print("4. Exit")
    print_line()


def main_menu():
    while True:
        show_welcome_screen()
        choice = get_menu_choice("Select an option (1-4): ", {1, 2, 3, 4})

        if choice == 1:
            analyse_customer_churn()
            pause()
        elif choice == 2:
            check_customer_risk()
            pause()
        elif choice == 3:
            help_menu()
        elif choice == 4:
            print("\nThank you for using TeleGuard: Telecom Retention & Churn Intelligence.")
            print("Exiting the system. Goodbye!")
            break


# ---------------------------------------------------------
# 4. ANALYSE CUSTOMER CHURN
# ---------------------------------------------------------

def analyse_customer_churn():
    """Run the existing analysis functions and save their normal outputs."""
    print("\n" + "=" * 68)
    print("                 CUSTOMER CHURN ANALYSIS")
    print("=" * 68)

    try:
        if not DATA_FILE.is_file():
            print("\nDataset not found.")
            print("Expected location:", DATA_FILE)
            print("Run synthetic-data-gen.py first.")
            return

        analysis = load_analysis_module()
        df = analysis.load_data()

        if df is None:
            print("\nThe dataset could not be loaded.")
            return

        if df.empty:
            print("\nThe dataset is empty. Generate the data again.")
            return

        required_columns = set(FEATURE_COLUMNS + [
            "customer_id", "total_charges", "churn"
        ])
        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            print("\nDataset is missing required columns:")
            print(", ".join(sorted(missing_columns)))
            return

        print("\nSTEP 1 — DATASET OVERVIEW")
        analysis.inspect_data(df)

        print("\nSTEP 2 — BUSINESS PERFORMANCE INDICATORS")
        analysis.calculate_kpis(df)

        print("\nSTEP 3 — CUSTOMER SEGMENTATION")
        analysis.segment_analysis(df)

        print("\nSTEP 4 — TRAIN AND EVALUATE LOGISTIC REGRESSION")
        final_model, X_test, y_test = analysis.train_churn_model(df)

        print("\nSTEP 5 — CUSTOMER-LEVEL RISK PROFILING")
        risk_df = analysis.generate_customer_risk_profiles(df, final_model)

        print("\nSTEP 6 — CHURN-ASSOCIATED FACTORS")
        importance_df = analysis.analyze_churn_drivers(
            final_model, X_test, y_test
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        risk_output = OUTPUT_DIR / "customer_risk_profiles.csv"
        importance_output = OUTPUT_DIR / "feature_importance.csv"

        risk_df.to_csv(risk_output, index=False)
        importance_df.to_csv(importance_output, index=False)

        print("\n" + "=" * 68)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 68)
        print("Customer risk profiles saved to:", risk_output)
        print("Feature importance saved to:", importance_output)
        print(
            "\nThese output files are also used by "
            "code-visualization.py."
        )

    except FileNotFoundError as error:
        print("\nFILE ERROR:", error)
        print("Check that the dataset and Python files are in the correct folders.")
    except ImportError as error:
        print("\nDEPENDENCY / IMPORT ERROR:", error)
        print("Install the dependencies listed in requirements.txt.")
    except (KeyError, ValueError) as error:
        print("\nDATA OR ANALYSIS ERROR:", error)
        print("Check the dataset columns, values and format.")
    except Exception as error:
        print("\nThe analysis could not be completed.")
        print("Reason:", error)
        print(
            "Check the error message and project files. "
            "The menu will remain available."
        )


# ---------------------------------------------------------
# 5. CHECK A CUSTOMER'S CHURN RISK
# ---------------------------------------------------------

def collect_customer_details():
    """Collect the exact model features and validate each value."""
    print("\nEnter the customer's details.")
    print("Use the displayed ranges and categories.")

    customer = {}

    # Collect numerical features in the same names used during training.
    for feature in NUMERICAL_FEATURES:
        minimum, maximum, number_type = NUMERICAL_RANGES[feature]
        customer[feature] = get_numeric_input(
            feature, minimum, maximum, number_type
        )

    # Collect categorical features in the same names used during training.
    for feature in CATEGORICAL_FEATURES:
        customer[feature] = get_category_input(
            feature, VALID_CATEGORIES[feature]
        )

    return customer


def check_customer_risk():
    """Predict churn probability using the existing trained pipeline."""
    print("\n" + "=" * 68)
    print("                INDIVIDUAL CUSTOMER RISK CHECK")
    print("=" * 68)
    print(
        "\nThis option estimates the churn risk of a hypothetical "
        "customer using the project's Logistic Regression model."
    )

    try:
        customer = collect_customer_details()

        # Build the input DataFrame in the exact feature order used in training.
        customer_df = pd.DataFrame(
            [[customer[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS
        )

        if not DATA_FILE.is_file():
            print("\nDataset not found:", DATA_FILE)
            print("Run synthetic-data-gen.py first.")
            return

        analysis = load_analysis_module()
        df = analysis.load_data()

        if df is None or df.empty:
            print("\nThe customer dataset could not be loaded or is empty.")
            return

        # Train using the existing project's preprocessing and model logic.
        # This intentionally reuses train_churn_model rather than creating
        # a second, inconsistent prediction pipeline.
        print("\nPreparing the project's Logistic Regression model...")
        final_model, _, _ = analysis.train_churn_model(df)

        probability = float(final_model.predict_proba(customer_df)[0, 1])

        if probability < 0.15:
            risk_category = "LOW"
            interpretation = (
                "The model estimates a comparatively lower likelihood "
                "of churn than the project's other risk bands."
            )
        elif probability < 0.30:
            risk_category = "MEDIUM"
            interpretation = (
                "The model estimates a moderate likelihood of churn. "
                "The customer may merit routine monitoring."
            )
        else:
            risk_category = "HIGH"
            interpretation = (
                "The model estimates a higher likelihood of churn. "
                "The customer may be prioritised for a retention review."
            )

        print("\n" + "=" * 68)
        print("                 CUSTOMER RISK RESULT")
        print("=" * 68)
        print(f"Estimated churn probability : {probability * 100:.2f}%")
        print(f"Predicted risk category     : {risk_category}")
        print("\nWhat this means:")
        print(interpretation)
        print(
            "\nThe probability is a model-based estimate from the "
            "synthetic project data.\nIt is not a guarantee that the "
            "customer will or will not leave."
        )
        print(
            "\nSuggested business use: treat this score as one input "
            "for prioritising\ncustomer-retention conversations, not "
            "as an automatic decision."
        )

    except (ValueError, KeyError) as error:
        print("\nINPUT / DATA ERROR:", error)
        print("Please check the entered details and try again.")
    except FileNotFoundError as error:
        print("\nFILE ERROR:", error)
        print("Check that the required project files are present.")
    except ImportError as error:
        print("\nDEPENDENCY / IMPORT ERROR:", error)
        print("Install the dependencies listed in requirements.txt.")
    except Exception as error:
        print("\nThe risk estimate could not be generated.")
        print("Reason:", error)
        print("Please check the project files and try again.")


# ---------------------------------------------------------
# 6. HELP & GUIDE
# ---------------------------------------------------------

def help_menu():
    while True:
        print("\n" + "=" * 68)
        print("                       HELP & GUIDE")
        print("=" * 68)
        print("1. How to Operate the System")
        print("2. Statistical Concepts Explained")
        print("3. Business Concepts Explained")
        print("4. Understanding Results and Risk Categories")
        print("5. Troubleshooting / Common Errors")
        print("6. Return to Main Menu")
        print_line()

        choice = get_menu_choice("Select a help topic (1-6): ", {1, 2, 3, 4, 5, 6})

        if choice == 1:
            help_operation()
        elif choice == 2:
            help_statistics()
        elif choice == 3:
            help_business()
        elif choice == 4:
            help_results()
        elif choice == 5:
            help_troubleshooting()
        elif choice == 6:
            return

        pause()


def help_operation():
    print("\nHOW TO OPERATE THE SYSTEM")
    print_line()
    print("MAIN MENU OPTIONS")
    print("1. Analyse Customer Churn")
    print(
        "   Runs the existing analysis script: dataset overview, business "
        "KPIs,\n   segment churn rates, Logistic Regression evaluation, "
        "customer risk\n   profiles and predictive feature importance. "
        "It saves the risk\n   profile and feature-importance CSV files "
        "in the output folder."
    )
    print("\n2. Check a Customer's Churn Risk")
    print(
        "   Enter one hypothetical customer's details. The system "
        "validates\n   the values, trains the same Logistic Regression "
        "pipeline used by\n   the project and estimates a churn "
        "probability and risk band."
    )
    print("\n3. Help & Guide")
    print("   Opens this built-in user guide.")
    print("\n4. Exit")
    print("   Closes the application.")
    print("\nCUSTOMER INPUTS")
    print(
        "   Numerical inputs: age, tenure, monthly charges, support "
        "tickets,\n   satisfaction, payment delays and usage hours."
    )
    print(
        "   Category inputs: city tier, contract type, internet service,\n"
        "   autopay, payment method and tech support plan."
    )
    print(
        "\nUse the allowed ranges and category options displayed by the "
        "program.\nMenu selections must be entered as numbers."
    )


def help_statistics():
    print("\nSTATISTICAL CONCEPTS EXPLAINED")
    print_line()
    concepts = [
        (
            "Churn rate",
            "The percentage of customers in the dataset who are marked "
            "as having left. It is calculated as churned customers divided "
            "by total customers, multiplied by 100."
        ),
        (
            "Probability",
            "A value between 0 and 1 (or 0% and 100%) representing the "
            "model's estimated likelihood of an outcome."
        ),
        (
            "Association / correlation",
            "A statistical relationship between variables. An association "
            "does not prove that one variable causes another."
        ),
        (
            "Logistic Regression",
            "A supervised learning method used for a yes/no outcome. "
            "Here it estimates the probability that a customer will churn."
        ),
        (
            "Classification",
            "Assigning an observation to a category. The model's probability "
            "can be converted into a churn / no-churn prediction using a "
            "chosen threshold."
        ),
        (
            "Classification threshold",
            "The probability cut-off used to convert a probability into "
            "a predicted class. The project's final evaluation uses 0.25."
        ),
        (
            "Confusion matrix",
            "A table comparing actual outcomes with predicted outcomes. "
            "It shows correct predictions and the two types of error."
        ),
        (
            "Accuracy",
            "The proportion of all predictions that are correct."
        ),
        (
            "Precision",
            "Among customers predicted to churn, the proportion who "
            "actually churned."
        ),
        (
            "Recall",
            "Among customers who actually churned, the proportion the "
            "model successfully identified."
        ),
        (
            "F1-score",
            "A combined measure of precision and recall. It is useful "
            "when both missed churners and false alarms matter."
        ),
        (
            "ROC-AUC",
            "A measure of how well the model ranks churners above "
            "non-churners across different classification thresholds. "
            "A value closer to 1 indicates stronger separation."
        ),
        (
            "Permutation importance",
            "A model interpretation method. It shuffles one feature and "
            "measures how much predictive performance falls. A larger "
            "drop suggests the model relies more on that feature."
        ),
        (
            "Training, validation and test data",
            "Training data fits the model; validation data helps compare "
            "choices such as thresholds; test data is held back for a "
            "final, independent performance check."
        ),
    ]

    for title, explanation in concepts:
        print(f"\n{title}:")
        print(explanation)


def help_business():
    print("\nBUSINESS CONCEPTS EXPLAINED")
    print_line()
    concepts = [
        (
            "Customer churn",
            "When a customer stops using or subscribing to a company's "
            "service."
        ),
        (
            "Customer retention",
            "Business activities intended to keep existing customers "
            "subscribed and satisfied."
        ),
        (
            "Customer tenure",
            "How long a customer has been with the telecom operator, "
            "measured here in months."
        ),
        (
            "Monthly charges",
            "The amount a customer is charged for telecom services in "
            "a typical month."
        ),
        (
            "Contract type",
            "The subscription commitment: month-to-month, one-year or "
            "two-year in this project."
        ),
        (
            "Support tickets",
            "Recorded customer requests or complaints handled by the "
            "support team."
        ),
        (
            "Customer satisfaction",
            "A rating of the customer's satisfaction, represented here "
            "on a scale from 1 (low) to 5 (high)."
        ),
        (
            "Autopay",
            "An arrangement in which a customer's payment is collected "
            "automatically rather than requiring a manual payment."
        ),
        (
            "Churn risk",
            "The model's estimated likelihood that a customer will churn. "
            "It is a prioritisation signal, not a certainty."
        ),
        (
            "Customer lifetime value",
            "An estimate of the value a customer may generate over the "
            "duration of the relationship. This project does not calculate "
            "a customer lifetime value figure."
        ),
        (
            "Retention action",
            "A business response such as contacting a customer, resolving "
            "a service issue or reviewing a suitable plan."
        ),
    ]

    for title, explanation in concepts:
        print(f"\n{title}:")
        print(explanation)


def help_results():
    print("\nUNDERSTANDING RESULTS AND RISK CATEGORIES")
    print_line()
    print("PREDICTED CHURN PROBABILITY")
    print(
        "   The percentage is the model's estimated likelihood of churn "
        "for the\n   customer profile entered. For example, 40% means "
        "the model estimates\n   a 0.40 probability; it does not mean "
        "the customer definitely will leave."
    )

    print("\nPROJECT RISK BANDS")
    print("   Low    : probability below 15%")
    print("   Medium : probability from 15% to below 30%")
    print("   High   : probability of 30% or more")
    print(
        "   These are project-defined descriptive bands. They are not "
        "universal\n   telecom-industry standards."
    )

    print("\nMODEL EVALUATION")
    print(
        "   Accuracy measures overall correctness. Precision indicates "
        "how many\n   flagged customers truly churned. Recall indicates "
        "how many actual\n   churners were identified. The confusion "
        "matrix shows the counts\n   behind these measures."
    )
    print(
        "\n   A false positive is a retained customer incorrectly flagged "
        "as likely\n   to churn. A false negative is a customer who "
        "churns but was not flagged."
    )

    print("\nVISUALISATIONS")
    print(
        "   The charts in output/charts show churn distribution, churn "
        "across\n   contract, tenure and satisfaction groups, customer "
        "risk distribution,\n   top predictive factors and the "
        "contract-tenure churn heatmap."
    )

    print("\nIMPORTANT LIMITATION")
    print(
        "   The dataset is synthetic. The model demonstrates an "
        "analytical\n   workflow and should not be treated as a "
        "production telecom\n   prediction system or as proof of "
        "causation."
    )


def help_troubleshooting():
    print("\nTROUBLESHOOTING / COMMON ERRORS")
    print_line()
    print("1. Dataset file not found")
    print(
        "   Confirm that synthetic_telecom_customers.csv is inside "
        "Code/data/.\n   Run synthetic-data-gen.py if the file has "
        "not been generated."
    )

    print("\n2. Analysis script not found")
    print(
        "   Keep menu.py and code-analysis.py together inside the "
        "same Code folder."
    )

    print("\n3. Missing Python packages")
    print(
        "   Open a terminal in the project folder and install the "
        "dependencies:\n   pip install -r requirements.txt"
    )

    print("\n4. Invalid menu selection")
    print(
        "   Enter one of the displayed whole-number options. The "
        "program will ask again if the entry is invalid."
    )

    print("\n5. Invalid customer details")
    print(
        "   Use the displayed numerical ranges and choose a category "
        "from the numbered list. Blank values are not accepted."
    )

    print("\n6. Model or analysis error")
    print(
        "   Check that the dataset has not been renamed or edited and "
        "that code-analysis.py is the supplied project version. "
        "Install the required packages and retry."
    )

    print("\n7. Visualisations are missing")
    print(
        "   First select Analyse Customer Churn to regenerate the "
        "risk-profile and feature-importance CSV files. Then run "
        "code-visualization.py from the Code folder."
    )

    print("\n8. Still facing an issue?")
    print(
        "   Close the program, check the folder structure and "
        "requirements.txt, then launch menu.py again."
    )


# ---------------------------------------------------------
# 7. APPLICATION ENTRY POINT
# ---------------------------------------------------------

def main():
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user. Goodbye.")
    except EOFError:
        print("\n\nNo further input received. Closing the application.")
    except Exception as error:
        print("\nAn unexpected application error occurred:", error)
        print("Please check the project files and try again.")


if __name__ == "__main__":
    main()
