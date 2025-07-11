import polars as pl
import time
from collections import Counter


def read_csv_polars(filepath):
    """Read CSV file using polars."""
    try:
        df = pl.read_csv(filepath, encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def analyze_column_polars(df, column_name):
    """Analyze a single column using polars methods."""
    # Get column and remove nulls
    series = df.select(pl.col(column_name)).to_series()
    series = series.filter(series.is_not_null())

    # Only filter empty strings for string columns
    if series.dtype == pl.Utf8:
        series = series.filter(series != "")

    if series.is_empty():
        return {'column': column_name, 'type': 'empty', 'stats': {}}

    # Check if column is numeric by checking dtype first
    if series.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64]:
        # Already numeric, no need to cast
        numeric_series = series.cast(pl.Float64)
        std_dev = numeric_series.std(ddof=1)
        stats = {
            'count': len(numeric_series),
            'mean': numeric_series.mean(),
            'min': numeric_series.min(),
            'max': numeric_series.max(),
            'std_dev': std_dev if std_dev is not None else 0.0
        }
        return {'column': column_name, 'type': 'numeric', 'stats': stats}

    # For non-numeric columns, try to cast to see if they contain numeric strings
    try:
        # Try to cast to float to check if numeric
        numeric_series = series.cast(pl.Float64, strict=False)
        numeric_series = numeric_series.filter(numeric_series.is_not_null())

        if len(numeric_series) == len(series):  # All values are numeric
            std_dev = numeric_series.std(ddof=1)
            stats = {
                'count': len(numeric_series),
                'mean': numeric_series.mean(),
                'min': numeric_series.min(),
                'max': numeric_series.max(),
                'std_dev': std_dev if std_dev is not None else 0.0
            }
            return {'column': column_name, 'type': 'numeric', 'stats': stats}
    except:
        pass

    # Categorical analysis
    value_counts = series.value_counts().sort("count", descending=True)
    most_common = []

    for i in range(min(3, len(value_counts))):
        row = value_counts.row(i)
        most_common.append((str(row[0]), row[1]))

    stats = {
        'total_count': len(series),
        'unique_count': series.n_unique(),
        'most_common': most_common
    }
    return {'column': column_name, 'type': 'categorical', 'stats': stats}


def print_column_analysis_polars(analysis):
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
        std_dev = stats['std_dev']
        if std_dev is not None:
            print(f"  Std Dev: {std_dev:.4f}")
        else:
            print(f"  Std Dev: N/A (insufficient data)")

    elif col_type == 'categorical':
        print(f"  Type: Categorical")
        print(f"  Total Count: {stats['total_count']}")
        print(f"  Unique Values: {stats['unique_count']}")
        print(f"  Most Common:")
        for value, count in stats['most_common']:
            print(f"    '{value}': {count} times")


def compute_stats_polars(df, group_cols=None):
    """Compute statistics for dataset using polars, optionally grouped by columns."""
    start_time = time.time()

    if group_cols:
        print(f"\n{'='*50}")
        print(f"GROUPED ANALYSIS BY: {', '.join(group_cols)}")
        print(f"{'='*50}")

        # Check if grouping columns exist
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Grouping columns not found: {missing_cols}")
            print("Available columns:", df.columns)
            return

        # Group data
        grouped = df.group_by(group_cols)

        # Get group keys for counting
        group_keys = df.select(group_cols).unique()
        print(f"Number of groups: {len(group_keys)}")

        # Analyze each group (limit output for readability)
        for i, group_key in enumerate(group_keys.iter_rows()):
            if i >= 5:  # Show only first 5 groups
                print(f"\n... and {len(group_keys) - 5} more groups")
                break

            # Filter data for this group
            filter_conditions = []
            for j, col in enumerate(group_cols):
                filter_conditions.append(pl.col(col) == group_key[j])

            # Combine conditions with AND
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition

            group_df = df.filter(combined_filter)

            print(f"\n--- Group {i+1}: {group_key} ---")
            print(f"Group size: {len(group_df)} rows")

            # Analyze each column in the group (excluding grouping columns)
            for col in group_df.columns:
                if col not in group_cols:
                    analysis = analyze_column_polars(group_df, col)
                    print_column_analysis_polars(analysis)
    else:
        print(f"\n{'='*50}")
        print("OVERALL DATASET ANALYSIS")
        print(f"{'='*50}")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")

        # Overall dataset info
        print(f"\nDataset Info:")
        print(f"  Estimated memory usage: {df.estimated_size('mb'):.2f} MB")
        print(f"  Data types: {dict(zip(df.columns, df.dtypes))}")

        # Analyze each column
        for col in df.columns:
            analysis = analyze_column_polars(df, col)
            print_column_analysis_polars(analysis)

        # Additional polars-specific insights
        print(f"\n{'='*30}")
        print("POLARS BUILT-IN DESCRIBE:")
        print(f"{'='*30}")

        # Get numeric columns
        numeric_cols = []
        for col in df.columns:
            if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                 pl.Float32, pl.Float64]:
                numeric_cols.append(col)

        if numeric_cols:
            print(f"\nNumeric columns summary:")
            try:
                numeric_df = df.select(numeric_cols)
                print(numeric_df.describe())
            except Exception as e:
                print(f"Error in describe: {e}")

        # Categorical columns info
        categorical_cols = [col for col in df.columns
                            if df[col].dtype in [pl.Utf8, pl.Categorical]]

        if categorical_cols:
            print(f"\nCategorical columns summary:")
            for col in categorical_cols:
                try:
                    unique_count = df[col].n_unique()
                    null_count = df[col].null_count()

                    # Get most frequent value
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 0:
                        most_frequent = value_counts.sort(
                            "count", descending=True).row(0)[0]
                    else:
                        most_frequent = 'N/A'

                    print(f"\n{col}:")
                    print(f"  Unique values: {unique_count}")
                    print(f"  Most frequent: {most_frequent}")
                    print(f"  Missing values: {null_count}")
                except Exception as e:
                    print(f"Error analyzing {col}: {e}")

    end_time = time.time()
    print(
        f"\n--- Analysis completed in {end_time - start_time:.4f} seconds ---")


def main():
    """Main function to run the polars analysis."""
    print("Polars Descriptive Statistics Analysis")
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
    df = read_csv_polars(filepath)

    if df is None:
        print("No data loaded. Exiting.")
        return

    # Perform analyses
    print(
        f"\nDataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")

    # Overall analysis
    compute_stats_polars(df)

    # Grouped analyses
    compute_stats_polars(df, group_cols=["page_id"])
    compute_stats_polars(df, group_cols=["page_id", "ad_id"])


if __name__ == "__main__":
    main()
