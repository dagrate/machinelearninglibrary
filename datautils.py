import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

# Ensure pandas options allow us to see the full reports
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

class DataQualityAnalyzer:
    """
    A comprehensive data quality analyzer built for binary classification 
    projects, especially in a risk modeling context.

    This class runs a pipeline of checks and prints a formatted report.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
        target_variable (str): The name of the target variable (e.g., 'is_default').
    """

    def __init__(self, df: pd.DataFrame, target_variable: str):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")
        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in DataFrame.")
            
        self.df = df.copy()  # Avoid modifying the original DataFrame
        self.target = target_variable
        
        # Automatically identify feature types, excluding the target
        all_cols = set(self.df.columns)
        all_cols.remove(self.target)
        
        self.numerical_features = self.df[list(all_cols)].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df[list(all_cols)].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # This dictionary will store all our analysis results
        self.summary: Dict[str, Any] = {}
        
        print(f"DataQualityAnalyzer initialized.")
        print(f"Total Observations: {len(self.df)}")
        print(f"Total Features: {len(self.df.columns) - 1}")
        print(f"Target Variable: '{self.target}'")
        print("-" * 80)

    def analyze_target_distribution(self) -> pd.DataFrame:
        """
        Crucial for binary classification: Check for class imbalance.
        """
        dist = self.df[self.target].value_counts()
        dist_percent = self.df[self.target].value_counts(normalize=True) * 100
        
        dist_df = pd.DataFrame({
            'Count': dist,
            'Percentage': dist_percent.round(2)
        })
        
        self.summary['target_distribution'] = dist_df
        return dist_df

    def analyze_missing_values(self) -> pd.DataFrame:
        """
        (Req 1) Count and percentage of missing values per feature.
        """
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percent': missing_percent.round(2)
        })
        
        # We only care about columns that *have* missing data
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
            by='missing_percent', ascending=False
        )
        
        self.summary['missing_values'] = missing_df
        return missing_df

    def analyze_numerical_summary(self) -> pd.DataFrame:
        """
        (Req 0) Get basic descriptive statistics for all numerical features.
        """
        if not self.numerical_features:
            self.summary['numerical_summary'] = pd.DataFrame()
            return pd.DataFrame()

        df_num = self.df[self.numerical_features]
        
        # .describe() gives us most of what we need
        stats = df_num.describe().T
        
        # Ensure all requested stats are present
        stats['median'] = df_num.median()
        stats['average'] = stats['mean'] # Add 'average' alias for clarity
        
        # Reorder and select final columns
        final_cols = ['count', 'average', 'std', 'min', 'median', 'max']
        stats = stats[final_cols]
        
        self.summary['numerical_summary'] = stats.round(3)
        return self.summary['numerical_summary']

    def analyze_categorical_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        (Req 2) Get unique values and their counts for categorical features.
        """
        cat_summary = {}
        if not self.categorical_features:
            self.summary['categorical_summary'] = cat_summary
            return cat_summary

        for col in self.categorical_features:
            n_unique = self.df[col].nunique()
            
            # Get top 5 most frequent values + their percentages
            top_5_values = (self.df[col].value_counts(normalize=True, dropna=False) * 100).head(5).round(2).to_dict()
            
            cat_summary[col] = {
                'unique_count': n_unique,
                'top_5_values_percent': top_5_values
            }
            
        self.summary['categorical_summary'] = cat_summary
        return cat_summary

    def check_cardinality_and_variance(self) -> pd.DataFrame:
        """
        (Req 4 - Expert Step) Check for problematic features:
        1. High Cardinality: Features with too many unique values (e.g., IDs).
        2. Zero Variance: Features that are constant (useless for modeling).
        """
        all_features = self.numerical_features + self.categorical_features
        variance_stats = []

        for col in all_features:
            n_unique = self.df[col].nunique(dropna=False)
            status = "OK"
            
            if n_unique == 1:
                status = "Zero Variance (Constant)"
            elif n_unique == len(self.df):
                status = "High Cardinality (Unique ID)"
            elif (n_unique > (len(self.df) * 0.7)) and (col in self.categorical_features):
                 status = "High Cardinality (Suspicious)"
            
            variance_stats.append({
                'feature': col,
                'unique_count': n_unique,
                'status': status
            })
            
        variance_df = pd.DataFrame(variance_stats).set_index('feature')
        
        # Filter to only show problematic features
        problematic_features = variance_df[variance_df['status'] != 'OK']
        
        self.summary['problematic_features'] = problematic_features
        return problematic_features

    def analyze_correlation(self, threshold: float = 0.9) -> pd.DataFrame:
        """
        (Req 4 - Expert Step) Check for high multicollinearity.
        This is crucial for linear models (like Logistic Regression) 
        which are common in risk for their interpretability.
        """
        if not self.numerical_features:
            self.summary['highly_correlated_pairs'] = pd.DataFrame()
            return pd.DataFrame()

        corr_matrix = self.df[self.numerical_features].corr()
        
        # Get upper triangle of the matrix (to avoid duplicates)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs with correlation above the threshold
        highly_correlated = upper_tri[upper_tri.abs() > threshold].stack().reset_index()
        highly_correlated.columns = ['Feature_1', 'Feature_2', 'Correlation']
        highly_correlated['Correlation'] = highly_correlated['Correlation'].round(3)
        
        self.summary['highly_correlated_pairs'] = highly_correlated
        return highly_correlated

    def _print_report(self):
        """
        A helper function to print the full analysis in a clean,
        auditable format.
        """
        print("\n" + "="*80)
        print("    🏦 DATA QUALITY & AUDIT REPORT 🏦")
        print("="*80)

        # --- 1. Target Distribution (Most Important) ---
        print("\n--- 1. Target Variable Distribution ---\n")
        print(self.summary['target_distribution'])
        if (self.summary['target_distribution']['Percentage'] < 10).any():
            print("\nWARNING: Target is highly imbalanced. Consider using stratification,")
            print("         SMOTE, or appropriate metrics (e.g., AUC-PR, F1-Score).")

        # --- 2. Missing Values ---
        print("\n--- 2. Missing Value Analysis ---\n")
        missing_df = self.summary['missing_values']
        if missing_df.empty:
            print("Congratulations! No missing values found.")
        else:
            print(missing_df)
            high_missing = missing_df[missing_df['missing_percent'] > 50].index.tolist()
            if high_missing:
                print(f"\nWARNING: Features with >50% missing data: {high_missing}")
                print("         These may need to be dropped or heavily engineered.")

        # --- 3. Problematic Features (Variance/Cardinality) ---
        print("\n--- 3. Feature Cardinality & Variance Check ---\n")
        problem_df = self.summary['problematic_features']
        if problem_df.empty:
            print("No zero-variance or high-cardinality ID features found.")
        else:
            print(problem_df)
            print("\nACTION: 'Zero Variance' features should be dropped.")
            print("        'High Cardinality' features (e.g., IDs) should be")
            print("        dropped or investigated as potential data leaks.")

        # --- 4. Numerical Summary ---
        print("\n--- 4. Numerical Features Summary ---\n")
        num_summary = self.summary['numerical_summary']
        if num_summary.empty:
            print("No numerical features found to analyze.")
        else:
            print(num_summary)

        # --- 5. Categorical Summary ---
        print("\n--- 5. Categorical Features Summary ---\n")
        cat_summary = self.summary['categorical_summary']
        if not cat_summary:
            print("No categorical features found to analyze.")
        else:
            print(f"Found {len(cat_summary)} categorical features.")
            # Print summary for first 5 features for brevity
            for i, (col, stats) in enumerate(cat_summary.items()):
                if i >= 5:
                    print(f"\n... and {len(cat_summary) - 5} more categorical features.")
                    break
                print(f"\n  Feature: '{col}'")
                print(f"    Unique Values: {stats['unique_count']}")
                print(f"    Top 5 Values (%): {stats['top_5_values_percent']}")

        # --- 6. Correlation Analysis ---
        print("\n--- 6. High Correlation Analysis (Numerical Features) ---\n")
        corr_df = self.summary['highly_correlated_pairs']
        if corr_df.empty:
            print("No highly correlated feature pairs found (Threshold > 0.9).")
        else:
            print(corr_df)
            print("\nWARNING: High multicollinearity detected. This can destabilize")
            print("         model coefficients (e.g., in Logistic Regression).")
            print("         Consider dropping one feature from each pair.")

        print("\n" + "="*80)
        print("    Analysis Complete.")
        print("="*80)

    def run_analysis(self, print_report: bool = True) -> Dict[str, Any]:
        """
        (Req 3) This is the main pipeline method.
        It executes all analysis steps in order and prints a summary report.
        
        Args:
            print_report (bool): If True, prints the formatted report to console.

        Returns:
            dict: A dictionary containing all analysis results.
        """
        print("Starting analysis pipeline...")
        
        # Run all analysis methods
        self.analyze_target_distribution()
        self.analyze_missing_values()
        self.check_cardinality_and_variance()
        self.analyze_numerical_summary()
        self.analyze_categorical_summary()
        self.analyze_correlation()
        
        if print_report:
            self._print_report()
            
        print("Analysis finished.")
        return self.summary
