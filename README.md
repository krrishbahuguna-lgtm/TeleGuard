# TeleGuard — Telecom Customer Churn Risk Profiler

TeleGuard is a menu-driven Python decision-support project that analyses customer churn patterns in a simulated telecom dataset. It uses Logistic Regression to estimate customer-level churn probabilities, assigns Low, Medium, or High risk categories, and generates visualisations to support retention decisions.

> **Data and interpretation:** The dataset is synthetic and independently generated for this academic project. The IBM Telco Customer Churn dataset was used only as a reference for variable selection and terminology. TeleGuard's results are not real-world telecom statistics, and statistical associations should not be interpreted as causal relationships.

## Project structure

```text
TeleGuard/
├── Code/
│   ├── synthetic-data-gen.py
│   ├── code-analysis.py
│   ├── code-visualization.py
│   ├── menu.py
│   ├── data/
│   │   └── synthetic_telecom_customers.csv
│   └── output/
│       ├── customer_risk_profiles.csv
│       ├── feature_importance.csv
│       └── charts/ (8 generated charts)
├── Project_Proposal.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

## Requirements

- Python 3.14 (as specified in the supplied project documentation; use the same Python installation for installing dependencies and running the project).
- pip

## Installation

From the repository root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, if `python` is not recognised, try:

```powershell
py -m pip install -r requirements.txt
```

## Run the project

Run these commands from the repository root, in this order:

1. Generate the synthetic dataset (optional; the included dataset is already provided):

   ```bash
   python Code/synthetic-data-gen.py
   ```

2. Run customer churn analysis:

   ```bash
   python Code/code-analysis.py
   ```

3. Generate the eight visualisations (after analysis):

   ```bash
   python Code/code-visualization.py
   ```

4. Launch the interactive menu:

   ```bash
   python Code/menu.py
   ```

Analysis outputs are saved under `Code/output/`; charts are saved under `Code/output/charts/`.

## Understanding risk scores

The model estimates a churn probability between 0 and 1. Project-defined risk bands are:

- **Low:** below 15%
- **Medium:** 15% to below 30%
- **High:** 30% or above

The model's classification threshold for evaluation is 0.25; this is separate from the risk-band boundaries. A probability is an estimate, not a guarantee that a customer will churn.

## Troubleshooting

- **Missing package:** run `python -m pip install -r requirements.txt` using the same Python interpreter used to launch the scripts.
- **Dataset not found:** run `python Code/synthetic-data-gen.py` from the repository root.
- **Analysis outputs missing:** run `python Code/code-analysis.py` before the visualisation script.
- **Script not found:** ensure your terminal is in the repository root.

## Project files

- `synthetic-data-gen.py` — generates and validates the synthetic customer dataset.
- `code-analysis.py` — data inspection, KPI analysis, Logistic Regression, risk profiling, and feature importance.
- `code-visualization.py` — generates the eight charts.
- `menu.py` — interactive menu for analysis, hypothetical customer risk estimation, and guidance.
- `Project_Proposal.pdf` — original project proposal.
