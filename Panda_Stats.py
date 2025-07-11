import pandas as pd
import numpy as np
import time
from collections import Counter


def read_csv_pandas(filepath):
    """Read CSV file using pandas."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def analyze_column_pandas(df, column_name):
    """Analyze a single column using pandas methods."""
    series = df[column_name].dropna()  # Remove NaN values
    series = series[series != ""]  # Remove empty strings

    if series.empty:
        return {'column': column_name, 'type': 'empty', 'stats': {}}

    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(series) or series.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().all():
        # Convert to numeric if needed
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce').dropna()

        stats = {
            'count': len(series),
            'mean': series.mean(),
            'min': series.min(),
            'max': series.max(),
            'std_dev': series.std(ddof=1)  # Sample standard deviation
        }
        return {'column': column_name, 'type': 'numeric', 'stats': stats}
    else:
        # Categorical analysis
        value_counts = series.value_counts()
        most_common = [(str(val), count)
                       for val, count in value_counts.head(3).items()]

        stats = {
            'total_count': len(series),
            'unique_count': series.nunique(),
            'most_common': most_common
        }
        return {'column': column_name, 'type': 'categorical', 'stats': stats}


def print_column_analysis_pandas(analysis):
    """Print formatted analysis for a single column."""
    col_name = analysis['column']
    col_type = analysis['type']
    stats = analysis['stats']

    print(f"\nColumn: {col_name}")

    if col_type == 'empty':
        print("  No data available")
        return

    if col_type == 'numeric':
        print(f"  Type: Numeric")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Min: {stats['min']}")
        print(f"  Max: {stats['max']}")
        print(f"  Std Dev: {stats['std_dev']:.4f}")

    elif col_type == 'categorical':
        print(f"  Type: Categorical")
        print(f"  Total Count: {stats['total_count']}")
        print(f"  Unique Values: {stats['unique_count']}")
        print(f"  Most Common:")
        for value, count in stats['most_common']:
            print(f"    '{value}': {count} times")


def compute_stats_pandas(df, group_cols=None):
    """Compute statistics for dataset using pandas, optionally grouped by columns."""
    start_time = time.time()

    if group_cols:
        print(f"\n{'='*50}")
        print(f"GROUPED ANALYSIS BY: {', '.join(group_cols)}")
        print(f"{'='*50}")

        # Check if grouping columns exist
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Grouping columns not found: {missing_cols}")
            print("Available columns:", list(df.columns))
            return

        # Group data
        grouped = df.groupby(group_cols)
        print(f"Number of groups: {grouped.ngroups}")

        # Analyze each group (limit output for readability)
        for i, (group_key, group_df) in enumerate(grouped):
            if i >= 5:  # Show only first 5 groups
                print(f"\n... and {grouped.ngroups - 5} more groups")
                break

            print(f"\n--- Group {i+1}: {group_key} ---")
            print(f"Group size: {len(group_df)} rows")

            # Analyze each column in the group (excluding grouping columns)
            for col in group_df.columns:
                if col not in group_cols:
                    analysis = analyze_column_pandas(group_df, col)
                    print_column_analysis_pandas(analysis)
    else:
        print(f"\n{'='*50}")
        print("OVERALL DATASET ANALYSIS")
        print(f"{'='*50}")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")

        # Overall dataset info
        print(f"\nDataset Info:")
        print(
            f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"  Data types: {df.dtypes.value_counts().to_dict()}")

        # Analyze each column
        for col in df.columns:
            analysis = analyze_column_pandas(df, col)
            print_column_analysis_pandas(analysis)

        # Additional pandas-specific insights
        print(f"\n{'='*30}")
        print("PANDAS BUILT-IN DESCRIBE:")
        print(f"{'='*30}")

        # Numeric columns describe
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nNumeric columns summary:")
            print(df[numeric_cols].describe())

        # Categorical columns info
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical columns summary:")
            for col in categorical_cols:
                print(f"\n{col}:")
                print(f"  Unique values: {df[col].nunique()}")
                print(
                    f"  Most frequent: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
                print(f"  Missing values: {df[col].isna().sum()}")

    end_time = time.time()
    print(
        f"\n--- Analysis completed in {end_time - start_time:.4f} seconds ---")


def main():
    """Main function to run the pandas analysis."""
    print("Pandas Descriptive Statistics Analysis")
    print("=" * 50)

    # File selection
    print("Choose dataset:")
    print("1: 2024_fb_ads_president_scored_anon.csv")
    print("2: 2024_fb_posts_president_scored_anon.csv")
    print("3: 2024_tw_posts_president_scored_anon.csv")

    choice = input("Enter choice (1/2/3): ").strip()

    file_map = {
        "1": "2024_fb_ads_president_scored_anon.csv",
        "2": "2024_fb_posts_president_scored_anon.csv",
        "3": "2024_tw_posts_president_scored_anon.csv"
    }

    filepath = file_map.get(choice, "2024_fb_ads_president_scored_anon.csv")
    print(f"\nLoading dataset: {filepath}")

    # Load data
    df = read_csv_pandas(filepath)

    if df is None:
        print("No data loaded. Exiting.")
        return

    # Perform analyses
    print(
        f"\nDataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")

    # Overall analysis
    compute_stats_pandas(df)

    # Grouped analyses
    compute_stats_pandas(df, group_cols=["page_id"])
    compute_stats_pandas(df, group_cols=["page_id", "ad_id"])


if __name__ == "__main__":
    main()
