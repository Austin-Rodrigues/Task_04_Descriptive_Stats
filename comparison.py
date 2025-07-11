import pandas as pd
import polars as pl
import csv
import math
import time
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Any, Tuple
import json


class StatsComparator:
    """
    A comprehensive comparison tool for validating that Pure Python, Pandas, and Polars
    implementations produce identical statistical results.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.results = {}
        self.performance = {}

    def load_data_pure_python(self) -> List[Dict]:
        """Load data using pure Python CSV reader."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            print(f"Error loading data with pure Python: {e}")
            return []

    def load_data_pandas(self) -> pd.DataFrame:
        """Load data using pandas."""
        try:
            return pd.read_csv(self.filepath, encoding='utf-8')
        except Exception as e:
            print(f"Error loading data with pandas: {e}")
            return pd.DataFrame()

    def load_data_polars(self) -> pl.DataFrame:
        """Load data using polars."""
        try:
            return pl.read_csv(self.filepath, encoding='utf-8')
        except Exception as e:
            print(f"Error loading data with polars: {e}")
            return pl.DataFrame()

    def is_float(self, value) -> bool:
        """Check if a value can be converted to float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def calculate_numeric_stats_python(self, numbers: List[float]) -> Dict:
        """Calculate numeric statistics using pure Python."""
        if not numbers:
            return {}

        count = len(numbers)
        mean = sum(numbers) / count
        min_val = min(numbers)
        max_val = max(numbers)

        if count > 1:
            variance = sum((x - mean) ** 2 for x in numbers) / (count - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0

        return {
            'count': count,
            'mean': round(mean, 6),
            'min': min_val,
            'max': max_val,
            'std_dev': round(std_dev, 6)
        }

    def calculate_categorical_stats_python(self, values: List[str]) -> Dict:
        """Calculate categorical statistics using pure Python."""
        if not values:
            return {}

        freq = Counter(values)
        unique_count = len(freq)
        total_count = len(values)
        most_common = freq.most_common(3)

        return {
            'total_count': total_count,
            'unique_count': unique_count,
            'most_common': most_common
        }

    def analyze_column_python(self, data: List[Dict], column_name: str) -> Dict:
        """Analyze a single column using pure Python."""
        values = [row[column_name] for row in data
                  if row[column_name] != "" and row[column_name] is not None]

        if not values:
            return {'column': column_name, 'type': 'empty', 'stats': {}}

        if all(self.is_float(v) for v in values):
            numbers = [float(v) for v in values]
            stats = self.calculate_numeric_stats_python(numbers)
            return {'column': column_name, 'type': 'numeric', 'stats': stats}
        else:
            stats = self.calculate_categorical_stats_python(values)
            return {'column': column_name, 'type': 'categorical', 'stats': stats}

    def analyze_column_pandas(self, df: pd.DataFrame, column_name: str) -> Dict:
        """Analyze a single column using pandas."""
        series = df[column_name].dropna()
        series = series[series != ""]

        if series.empty:
            return {'column': column_name, 'type': 'empty', 'stats': {}}

        # Check if numeric
        if pd.api.types.is_numeric_dtype(series) or series.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().all():
            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors='coerce').dropna()

            stats = {
                'count': len(series),
                'mean': round(series.mean(), 6),
                'min': series.min(),
                'max': series.max(),
                'std_dev': round(series.std(ddof=1), 6)
            }
            return {'column': column_name, 'type': 'numeric', 'stats': stats}
        else:
            value_counts = series.value_counts()
            most_common = [(str(val), count)
                           for val, count in value_counts.head(3).items()]

            stats = {
                'total_count': len(series),
                'unique_count': series.nunique(),
                'most_common': most_common
            }
            return {'column': column_name, 'type': 'categorical', 'stats': stats}

    def analyze_column_polars(self, df: pl.DataFrame, column_name: str) -> Dict:
        """Analyze a single column using polars."""
        series = df.select(pl.col(column_name)).to_series()
        series = series.filter(series.is_not_null())

        if series.dtype == pl.Utf8:
            series = series.filter(series != "")

        if series.is_empty():
            return {'column': column_name, 'type': 'empty', 'stats': {}}

        # Check if numeric
        if series.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                            pl.Float32, pl.Float64]:
            numeric_series = series.cast(pl.Float64)
            std_dev = numeric_series.std(ddof=1)
            stats = {
                'count': len(numeric_series),
                'mean': round(numeric_series.mean(), 6),
                'min': numeric_series.min(),
                'max': numeric_series.max(),
                'std_dev': round(std_dev if std_dev is not None else 0.0, 6)
            }
            return {'column': column_name, 'type': 'numeric', 'stats': stats}

        # Try to cast to numeric
        try:
            numeric_series = series.cast(pl.Float64, strict=False)
            numeric_series = numeric_series.filter(
                numeric_series.is_not_null())

            if len(numeric_series) == len(series):
                std_dev = numeric_series.std(ddof=1)
                stats = {
                    'count': len(numeric_series),
                    'mean': round(numeric_series.mean(), 6),
                    'min': numeric_series.min(),
                    'max': numeric_series.max(),
                    'std_dev': round(std_dev if std_dev is not None else 0.0, 6)
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

    def run_overall_analysis(self) -> Dict[str, Dict]:
        """Run overall analysis with all three methods and measure performance."""
        print("=" * 60)
        print("RUNNING COMPREHENSIVE COMPARISON")
        print("=" * 60)

        results = {}

        # Pure Python Analysis
        print("\n1. Running Pure Python Analysis...")
        start_time = time.time()
        python_data = self.load_data_pure_python()
        if python_data:
            columns = python_data[0].keys()
            python_results = {}
            for col in columns:
                python_results[col] = self.analyze_column_python(
                    python_data, col)
        else:
            python_results = {}
        python_time = time.time() - start_time
        results['python'] = python_results
        self.performance['python'] = python_time
        print(f"   Completed in {python_time:.4f} seconds")

        # Pandas Analysis
        print("\n2. Running Pandas Analysis...")
        start_time = time.time()
        pandas_data = self.load_data_pandas()
        if not pandas_data.empty:
            pandas_results = {}
            for col in pandas_data.columns:
                pandas_results[col] = self.analyze_column_pandas(
                    pandas_data, col)
        else:
            pandas_results = {}
        pandas_time = time.time() - start_time
        results['pandas'] = pandas_results
        self.performance['pandas'] = pandas_time
        print(f"   Completed in {pandas_time:.4f} seconds")

        # Polars Analysis
        print("\n3. Running Polars Analysis...")
        start_time = time.time()
        polars_data = self.load_data_polars()
        if not polars_data.is_empty():
            polars_results = {}
            for col in polars_data.columns:
                polars_results[col] = self.analyze_column_polars(
                    polars_data, col)
        else:
            polars_results = {}
        polars_time = time.time() - start_time
        results['polars'] = polars_results
        self.performance['polars'] = polars_time
        print(f"   Completed in {polars_time:.4f} seconds")

        self.results = results
        return results

    def compare_results(self) -> Dict[str, bool]:
        """Compare results between all three implementations."""
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)

        comparison_results = {}

        if not all(method in self.results for method in ['python', 'pandas', 'polars']):
            print("ERROR: Not all methods have results to compare!")
            return {}

        # Get common columns
        python_cols = set(self.results['python'].keys())
        pandas_cols = set(self.results['pandas'].keys())
        polars_cols = set(self.results['polars'].keys())

        common_cols = python_cols.intersection(
            pandas_cols).intersection(polars_cols)

        print(f"Comparing {len(common_cols)} common columns...")

        overall_match = True

        for col in sorted(common_cols):
            python_result = self.results['python'][col]
            pandas_result = self.results['pandas'][col]
            polars_result = self.results['polars'][col]

            # Compare types
            types_match = (
                python_result['type'] == pandas_result['type'] == polars_result['type'])

            if not types_match:
                print(f"\n❌ MISMATCH in {col} - Types don't match:")
                print(f"   Python: {python_result['type']}")
                print(f"   Pandas: {pandas_result['type']}")
                print(f"   Polars: {polars_result['type']}")
                comparison_results[col] = False
                overall_match = False
                continue

            # Compare statistics
            if python_result['type'] == 'numeric':
                stats_match = self.compare_numeric_stats(
                    python_result['stats'],
                    pandas_result['stats'],
                    polars_result['stats']
                )
            elif python_result['type'] == 'categorical':
                stats_match = self.compare_categorical_stats(
                    python_result['stats'],
                    pandas_result['stats'],
                    polars_result['stats']
                )
            else:  # empty
                stats_match = True

            if stats_match:
                print(f"✅ {col} - All implementations match")
            else:
                print(f"❌ {col} - Statistics don't match")
                overall_match = False

            comparison_results[col] = stats_match

        # Overall summary
        print(f"\n" + "=" * 60)
        if overall_match:
            print("🎉 SUCCESS: All implementations produce identical results!")
        else:
            print("⚠️  WARNING: Some discrepancies found between implementations!")

        return comparison_results

    def compare_numeric_stats(self, python_stats: Dict, pandas_stats: Dict, polars_stats: Dict) -> bool:
        """Compare numeric statistics with tolerance for floating-point differences."""
        tolerance = 1e-6

        for key in ['count', 'mean', 'min', 'max', 'std_dev']:
            if key not in python_stats or key not in pandas_stats or key not in polars_stats:
                continue

            python_val = python_stats[key]
            pandas_val = pandas_stats[key]
            polars_val = polars_stats[key]

            if key == 'count':
                if not (python_val == pandas_val == polars_val):
                    print(
                        f"      Count mismatch: Python={python_val}, Pandas={pandas_val}, Polars={polars_val}")
                    return False
            else:
                if not (abs(python_val - pandas_val) < tolerance and
                        abs(pandas_val - polars_val) < tolerance):
                    print(
                        f"      {key} mismatch: Python={python_val}, Pandas={pandas_val}, Polars={polars_val}")
                    return False

        return True

    def compare_categorical_stats(self, python_stats: Dict, pandas_stats: Dict, polars_stats: Dict) -> bool:
        """Compare categorical statistics."""
        # Check counts
        if not (python_stats['total_count'] == pandas_stats['total_count'] == polars_stats['total_count']):
            print(
                f"      Total count mismatch: Python={python_stats['total_count']}, Pandas={pandas_stats['total_count']}, Polars={polars_stats['total_count']}")
            return False

        if not (python_stats['unique_count'] == pandas_stats['unique_count'] == polars_stats['unique_count']):
            print(
                f"      Unique count mismatch: Python={python_stats['unique_count']}, Pandas={pandas_stats['unique_count']}, Polars={polars_stats['unique_count']}")
            return False

        # Most common values might be in different order but should have same content
        python_common = set(python_stats['most_common'])
        pandas_common = set(pandas_stats['most_common'])
        polars_common = set(polars_stats['most_common'])

        if not (python_common == pandas_common == polars_common):
            print(f"      Most common values mismatch:")
            print(f"        Python: {python_stats['most_common']}")
            print(f"        Pandas: {pandas_stats['most_common']}")
            print(f"        Polars: {polars_stats['most_common']}")
            return False

        return True

    def performance_analysis(self):
        """Analyze and report performance differences."""
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)

        if not self.performance:
            print("No performance data available!")
            return

        # Sort by performance
        sorted_performance = sorted(
            self.performance.items(), key=lambda x: x[1])

        print(f"{'Method':<15} {'Time (seconds)':<15} {'Relative Speed':<15}")
        print("-" * 45)

        fastest_time = sorted_performance[0][1]

        for method, time_taken in sorted_performance:
            relative_speed = time_taken / fastest_time
            print(f"{method:<15} {time_taken:<15.4f} {relative_speed:<15.2f}x")

        # Performance recommendations
        print(
            f"\n🏆 Fastest: {sorted_performance[0][0]} ({sorted_performance[0][1]:.4f}s)")
        print(
            f"🐌 Slowest: {sorted_performance[-1][0]} ({sorted_performance[-1][1]:.4f}s)")

        speedup = sorted_performance[-1][1] / sorted_performance[0][1]
        print(f"⚡ Speedup: {speedup:.2f}x faster")

    def generate_report(self) -> str:
        """Generate a comprehensive comparison report."""
        report = []
        report.append("=" * 80)
        report.append(
            "COMPREHENSIVE STATISTICS IMPLEMENTATION COMPARISON REPORT")
        report.append("=" * 80)

        report.append(f"\nDataset: {self.filepath}")
        report.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Performance Summary
        report.append("\n📊 PERFORMANCE SUMMARY")
        report.append("-" * 40)
        if self.performance:
            sorted_perf = sorted(self.performance.items(), key=lambda x: x[1])
            fastest = sorted_perf[0]
            slowest = sorted_perf[-1]

            report.append(
                f"Fastest Implementation: {fastest[0]} ({fastest[1]:.4f}s)")
            report.append(
                f"Slowest Implementation: {slowest[0]} ({slowest[1]:.4f}s)")
            report.append(
                f"Performance Difference: {slowest[1]/fastest[1]:.2f}x")

        # Accuracy Summary
        report.append("\n🎯 ACCURACY SUMMARY")
        report.append("-" * 40)
        if hasattr(self, 'comparison_results'):
            total_cols = len(self.comparison_results)
            matching_cols = sum(
                1 for match in self.comparison_results.values() if match)
            accuracy = matching_cols / total_cols * 100 if total_cols > 0 else 0

            report.append(f"Total Columns Compared: {total_cols}")
            report.append(f"Matching Results: {matching_cols}")
            report.append(f"Accuracy: {accuracy:.1f}%")

        return "\n".join(report)

    def run_full_comparison(self):
        """Run the complete comparison workflow."""
        print("Starting comprehensive comparison of statistics implementations...")

        # Run analyses
        self.run_overall_analysis()

        # Compare results
        self.comparison_results = self.compare_results()

        # Performance analysis
        self.performance_analysis()

        # Generate final report
        print("\n" + self.generate_report())

        # Save results to file
        self.save_results()

    def save_results(self):
        """Save detailed results to JSON file for further analysis."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"comparison_results_{timestamp}.json"

        output_data = {
            'timestamp': timestamp,
            'dataset': self.filepath,
            'performance': self.performance,
            'results': self.results,
            'comparison': getattr(self, 'comparison_results', {})
        }

        try:
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main function to run the comparison."""
    print("Descriptive Statistics Implementation Comparison Tool")
    print("=" * 60)

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

    print(f"\nSelected dataset: {filepath}")

    # Initialize comparator
    comparator = StatsComparator(filepath)

    # Run full comparison
    comparator.run_full_comparison()

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
