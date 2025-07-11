# Task_04_Descriptive_Stats

## Descriptive Statistics with and without 3rd Party Libraries (Pandas/Polars)

This project implements and compares three approaches to perform descriptive statistics on datasets from the 2024 U.S. presidential election social media activity. The analysis is done using:

- Pure Python (standard library only)
- Pandas
- Polars

The project includes performance benchmarking, accuracy comparison, and observations on ease of use and implementation differences.

---

## Project Structure

```
├── comparison.py              # Master script comparing all 3 implementations
├── Pure_Python.py             # Descriptive stats using only the Python standard library
├── Panda_Stats.py             # Descriptive stats using Pandas
├── Polars_Stats.py            # Descriptive stats using Polars
├── README.md                  # Project overview and instructions
└── *.csv                      # Not included
```

---

## Dataset Download

Datasets are related to the 2024 U.S. Presidential Election and include Facebook ads, Facebook posts, and Twitter posts.

Files:
- 2024_fb_ads_president_scored_anon.csv
- 2024_fb_posts_president_scored_anon.csv
- 2024_tw_posts_president_scored_anon.csv

---

## How to Run

Install dependencies using pip:

```
pip install pandas polars numpy
```

### Run All Comparisons

Run the main comparison script which evaluates all three implementations on the selected dataset:

```
python comparison.py
```

Follow the prompt to choose a dataset.

### Run Individual Analysis Scripts

Each script can be executed independently for detailed analysis:

```
python Pure_Python.py
python Panda_Stats.py
python Polars_Stats.py
```

Each performs:
- Overall descriptive statistics
- Grouped analysis by `page_id`
- Grouped analysis by `page_id` and `ad_id` (or relevant columns)

---

## Summary of Results

### Accuracy Comparison

| Dataset                                 | Columns Compared | Matches | Accuracy |
|-----------------------------------------|------------------|---------|----------|
| 2024_fb_ads_president_scored_anon.csv   | 41               | 40      | 97.6%    |
| 2024_fb_posts_president_scored_anon.csv | 56               | 52      | 92.9%    |
| 2024_tw_posts_president_scored_anon.csv | 47               | 39      | 83.0%    |

Differences were primarily due to type mismatches and differences in value parsing or ordering.

### Performance Comparison

| Method  | Ads (s) | FB Posts (s) | Twitter Posts (s) |
|---------|---------|--------------|-------------------|
| Polars  | 3.31    | 0.27         | 0.22              |
| Python  | 10.36   | 0.71         | 1.00              |
| Pandas  | 23.40   | 1.44         | 1.51              |

Polars was consistently the fastest across all datasets. Pandas was the slowest but offered the most user-friendly API.

---

## Reflections

- Pandas is intuitive and widely used, but slower on large datasets.
- Polars provides excellent performance and is well-suited for large-scale processing.
- Pure Python is valuable for understanding the fundamentals but not practical for large datasets.
- Matching results across all three methods required careful handling of nulls, data types, and value coercion.

Recommendation:
- Use Pandas for ease of development and readability.
- Use Polars for performance-critical applications.
- Use Pure Python for educational purposes or low-level data handling.
