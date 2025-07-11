import csv
import math
import time
from collections import defaultdict, Counter


def is_float(value):
    """Check if a value can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def read_csv(filepath):
    """Read CSV file and return list of dictionaries."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def calculate_numeric_stats(numbers):
    """Calculate comprehensive numeric statistics."""
    if not numbers:
        return {}

    count = len(numbers)
    mean = sum(numbers) / count
    min_val = min(numbers)
    max_val = max(numbers)

    # Calculate standard deviation (sample standard deviation)
    if count > 1:
        variance = sum((x - mean) ** 2 for x in numbers) / (count - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    return {
        'count': count,
        'mean': mean,
        'min': min_val,
        'max': max_val,
        'std_dev': std_dev
    }


def calculate_categorical_stats(values):
    """Calculate comprehensive categorical statistics."""
    if not values:
        return {}

    freq = Counter(values)
    unique_count = len(freq)
    total_count = len(values)
    most_common = freq.most_common(3)  # Top 3 most common

    return {
        'total_count': total_count,
        'unique_count': unique_count,
        'most_common': most_common
    }


def analyze_column(data, column_name):
    """Analyze a single column and return statistics."""
    # Extract non-empty values
    values = [row[column_name] for row in data
              if row[column_name] != "" and row[column_name] is not None]

    if not values:
        return {'column': column_name, 'type': 'empty', 'stats': {}}

    # Check if all values are numeric
    if all(is_float(v) for v in values):
        numbers = [float(v) for v in values]
        stats = calculate_numeric_stats(numbers)
        return {'column': column_name, 'type': 'numeric', 'stats': stats}
    else:
        stats = calculate_categorical_stats(values)
        return {'column': column_name, 'type': 'categorical', 'stats': stats}


def print_column_analysis(analysis):
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


def compute_stats(data, group_cols=None):
    """Compute statistics for dataset, optionally grouped by columns."""
    start_time = time.time()

    if group_cols:
        print(f"\n{'='*50}")
        print(f"GROUPED ANALYSIS BY: {', '.join(group_cols)}")
        print(f"{'='*50}")

        # Group data
        grouped = defaultdict(list)
        for row in data:
            key = tuple(row[col] for col in group_cols)
            grouped[key].append(row)

        print(f"Number of groups: {len(grouped)}")

        # Analyze each group (limit output for readability)
        for i, (key, group_data) in enumerate(grouped.items()):
            if i >= 5:  # Show only first 5 groups to avoid overwhelming output
                print(f"\n... and {len(grouped) - 5} more groups")
                break

            print(f"\n--- Group {i+1}: {key} ---")
            print(f"Group size: {len(group_data)} rows")

            if group_data:
                columns = group_data[0].keys()
                for col in columns:
                    if col not in group_cols:  # Skip grouping columns
                        analysis = analyze_column(group_data, col)
                        print_column_analysis(analysis)
    else:
        print(f"\n{'='*50}")
        print("OVERALL DATASET ANALYSIS")
        print(f"{'='*50}")
        print(f"Total rows: {len(data)}")

        if data:
            columns = data[0].keys()
            print(f"Total columns: {len(columns)}")

            for col in columns:
                analysis = analyze_column(data, col)
                print_column_analysis(analysis)

    end_time = time.time()
    print(
        f"\n--- Analysis completed in {end_time - start_time:.4f} seconds ---")


def main():
    """Main function to run the analysis."""
    print("Pure Python Descriptive Statistics Analysis")
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
    data = read_csv(filepath)

    if not data:
        print("No data loaded. Exiting.")
        return

    # Perform analyses
    print(f"\nDataset loaded successfully: {len(data)} rows")

    # Overall analysis
    compute_stats(data)

    # Grouped analyses
    compute_stats(data, group_cols=["page_id"])
    compute_stats(data, group_cols=["page_id", "ad_id"])


if __name__ == "__main__":
    main()
