# 0.5 — NumPy & Pandas

> The two libraries you will touch every single day of your ML career. Master them before touching PyTorch.

---

## 1. Intuition first

- **NumPy** = fast, typed, N-dimensional arrays with vectorized math. Think "MATLAB in Python, but faster and free."
- **Pandas** = tabular data with row/column labels. Think "SQL + Excel + a scripting language, in memory."

Everything ML tabular / preprocessing / feature-engineering happens in these two. Deep-learning tensor libraries (PyTorch, JAX, TensorFlow) copy NumPy's API deliberately.

## 2. Why the topic exists

Python `for` loops are ~100× slower than C. NumPy pushes numeric loops into optimized C/Fortran (BLAS, LAPACK). Pandas builds an ergonomic labeled-array on top of NumPy.

## 3. What problem it solves

- Represent datasets efficiently (contiguous typed arrays vs Python lists of Python floats).
- Vectorized computation (one line = a whole loop).
- Alignment by labels (join, merge, groupby, resample) — the bread & butter of data wrangling.

## 4. Mathematics

Same as linear algebra. Pandas adds relational-algebra operations (project, filter, join, group-by-aggregate).

## 5. Every "formula" explained — key mental models

- **Broadcasting rules**: two arrays are compatible if for each trailing dimension either the sizes match, or one is 1, or one is missing.
- **Vectorization**: express computations as array ops so the Python interpreter runs once per array, not per element.
- **View vs copy**: slicing returns a view (shares memory) unless you use fancy indexing.

## 6. Every "variable" explained

| Symbol | Meaning |
|--------|---------|
| `ndarray.shape` | tuple of dimensions |
| `ndarray.dtype` | element type |
| `ndarray.strides` | bytes between consecutive elements in each axis |
| `DataFrame.index` | row labels |
| `DataFrame.columns` | column labels |

## 7. Step-by-step algorithm — a proper data workflow

1. `pd.read_csv/parquet` → DataFrame.
2. Inspect: `df.head()`, `df.info()`, `df.describe()`, `df.isna().sum()`.
3. Clean: dtype casting, missing-value strategy (drop / impute / flag).
4. Feature-engineer: derived columns, encodings, aggregations.
5. Convert to NumPy for modeling: `X = df[feature_cols].to_numpy()`.

## 8. Simple example

```python
import numpy as np
x = np.array([1, 2, 3, 4])
y = x ** 2 + 1        # array([2, 5, 10, 17]) — no loop
```

## 9. Real-world example

```python
import pandas as pd

sales = pd.read_parquet("sales.parquet")

monthly = (
    sales
      .assign(month=lambda d: d["date"].dt.to_period("M"))
      .groupby(["month", "region"], as_index=False)
      .agg(revenue=("amount", "sum"),
           orders=("order_id", "nunique"))
      .sort_values(["region", "month"])
)
```

## 10. Diagram

```
Raw sources                    Wrangled                       Modeling
+-----------+    read_*      +-----------+     to_numpy    +-----------+
| CSV/JSON  | -----------> | DataFrame | ---------------> |  ndarray  |
| Parquet   |               | (labeled) |                  |  (dense)  |
| DB / API  |               +-----------+                  +-----------+
+-----------+                                                    |
                                                                 v
                                                            scikit-learn
                                                              PyTorch
```

## 11. Implementation from scratch — a tiny NumPy-like class

```python
class Array:
    def __init__(self, data):
        self.data = data                 # nested list
        self.shape = self._shape(data)

    @staticmethod
    def _shape(d):
        s = []
        while isinstance(d, list):
            s.append(len(d))
            d = d[0] if d else 0
        return tuple(s)

    def flatten(self):
        def go(d):
            if isinstance(d, list):
                for v in d:
                    yield from go(v)
            else:
                yield d
        return list(go(self.data))

    def __add__(self, other):
        assert self.shape == other.shape
        return Array([a + b for a, b in zip(self.flatten(), other.flatten())])
```

## 12. Implementation using libraries

NumPy cheatsheet:

```python
import numpy as np

a = np.arange(12).reshape(3, 4)        # (3,4)
a.T                                    # transpose
a[:, ::2]                              # every other column
a.sum(axis=0)                          # column sums
a.mean(axis=1, keepdims=True)          # row means
np.where(a > 5, a, 0)                  # conditional
np.linalg.norm(a, axis=1)              # row norms
np.einsum("ij,jk->ik", a, b)           # explicit contraction
np.random.default_rng(42).normal(size=(1000, 5))
```

Pandas cheatsheet:

```python
import pandas as pd

df = pd.read_csv("data.csv", parse_dates=["ts"])
df["age_bucket"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 120])
df.groupby("age_bucket", observed=True)["spend"].agg(["mean", "std", "count"])
df.merge(users, on="user_id", how="left")
df.pivot_table(index="region", columns="month", values="revenue", aggfunc="sum")
df.rolling(window=7, on="date")["value"].mean()
```

## 13. Time complexity

- Vectorized array op on $n$ elements: O(n) in C — often 50–200× faster than Python loop of same big-O.
- `groupby(...).agg(...)`: O(n log n) worst case; O(n) with hash grouping.
- `merge` (hash-join): O(n + m).

## 14. Space complexity

- Arrays: `n * dtype.itemsize` bytes contiguous.
- DataFrames: sum of column sizes + object-column overhead (strings are Python objects unless you use `string[pyarrow]`).

## 15. Advantages

- Blazing fast.
- Rich ecosystem — every ML library speaks NumPy arrays.
- Pandas is the *de facto* interchange format for tabular data in Python.

## 16. Disadvantages

- Pandas eats memory (Python-object columns, no schema at load).
- No lazy evaluation → OOM on big data. (Solution: Polars, DuckDB, Dask.)
- Confusing API (`.iloc` vs `.loc`, `SettingWithCopyWarning`).

## 17. Interview questions

1. What is broadcasting? Give three examples.
2. Difference between `apply`, `map`, and vectorized ops in Pandas.
3. `.loc` vs `.iloc`?
4. Explain `SettingWithCopyWarning` and how to fix it.
5. What is a NumPy view vs copy?
6. When does Pandas use `object` dtype and why is that slow?
7. How does `groupby(...).transform(...)` differ from `.agg(...)`?
8. How would you handle a dataset larger than RAM?
9. Explain `merge` join types.
10. What is `np.einsum`? Show it doing a batched matrix multiply.

## 18. Common mistakes

- Iterating with `df.iterrows()` (10–1000× slower than vectorization).
- Chained assignment: `df[df.a > 0]["b"] = 1` → silently drops.
- Using `apply` when a vectorized op exists.
- Not setting `dtype` when reading big CSVs — Pandas guesses badly.
- Confusing `NaN` (float) with `None` (object).

## 19. Optimization techniques

- Use `pd.read_csv(..., dtype=..., usecols=...)`.
- Prefer `category` dtype for low-cardinality strings.
- Use `numexpr` / `polars` / `duckdb` for heavy ops.
- Chunked reading with `chunksize=` for streaming.
- Store intermediate results in Parquet, not CSV.

## 20. Coding exercises

1. Given a `(n, 3)` array of RGB pixels, convert to grayscale with $0.299R + 0.587G + 0.114B$ vectorized.
2. Implement one-hot encoding without a library.
3. In Pandas, compute a 30-day rolling median of stock returns per ticker.
4. Given transaction data, compute the fraction of new users per week.
5. Implement `groupby.mean` from scratch on a NumPy array + labels.
6. Deduplicate near-duplicate rows using vectorized cosine similarity.

## 21. Mini project

**Titanic EDA notebook** using only NumPy, Pandas, Matplotlib — produce cleaned features + basic summary stats + 6 informative plots.

## 22. Medium project

**E-commerce cohort analysis** — from a raw order log, produce a weekly retention heatmap and average LTV per acquisition cohort, all in vectorized Pandas.

## 23. Advanced project

**5-GB Parquet ETL** — build a Python pipeline that reads a multi-file Parquet dataset with Pandas + PyArrow, does complex windowed aggregations, and writes partitioned Parquet output. Beat a naive implementation by 5× using chunking, `pyarrow.dataset`, and categorical dtypes.

## 24. Where it is used in industry

Every data / ML team uses Pandas for prototyping and NumPy for numeric internals. Feature pipelines, model training scripts, offline batch jobs — all Pandas + NumPy.

## 25. How companies use it

- Feature stores (Feast, Tecton) speak Pandas.
- Airflow / Prefect tasks are often Pandas scripts.
- Notebooks at Netflix, Airbnb, Uber, Stripe: 90% Pandas.

## 26. When NOT to use it

- Data > 10 × RAM: use Spark, Dask, Polars, or DuckDB.
- Sub-millisecond latency needs: precompute and cache; don't call Pandas in the hot path.
- Distributed streaming: use Kafka + Flink / Beam.
