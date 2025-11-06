# --- Create a Sample Bank Loan DataFrame ---
# This data is intentionally "messy" to test all our checks.

data = {
    'customer_id': [f'C_{i}' for i in range(100)],  # High cardinality
    'loan_type': ['Mortgage', 'Personal', 'Auto', 'Mortgage', 'Personal'] * 20, # Categorical
    'loan_amount': np.random.normal(200000, 50000, 100).astype(int), # Numerical
    'age': [np.nan, 34, 55, 29, 62] * 20, # Numerical with missing
    'credit_score': np.random.randint(500, 800, 100), # Numerical
    'internal_rating': np.nan, # All missing
    'bank_branch': 'Main St Branch', # Zero variance
    'is_default': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] * 10 # Imbalanced target
}

# Add a highly correlated feature
data['credit_score_copy'] = data['credit_score'] + np.random.normal(0, 2, 100)

sample_df = pd.DataFrame(data)

# Make the 'internal_rating' column fully null, except for one value
sample_df.loc[0, 'internal_rating'] = 'A' # Now it's not *all* null, but >90% missing


# --- 1. Initialize the Analyzer ---
# Provide the DataFrame and the name of the target variable.
analyzer = DataQualityAnalyzer(df=sample_df, target_variable='is_default')

# --- 2. Run the Full Analysis Pipeline ---
# This one command runs all checks and prints the complete report.
analysis_results = analyzer.run_analysis()
