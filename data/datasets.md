# Datasets

Place the three CSV files below in this folder, under exactly these names.
`src/data.py` drops columns with more than 40% missing values, keeps the
numeric non-sensitive columns, median-imputes, standardizes, and draws a
subsample stratified by the sensitive attribute.

| File | Dataset | Records | Sensitive attribute | Source |
|---|---|---:|---|---|
| `diabetic_data.csv` | Diabetes 130-US hospitals (1999–2008) | 101,766 | `gender` (Male / Female) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) — file `diabetic_data.csv` inside the archive |
| `uci_census.csv` | UCI Census (Adult), train + test | 48,842 | `gender` (Male / Female) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult); a single CSV with a header row and the column names `age, workclass, fnlwgt, education, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country, income` (e.g. the widely used `adult.csv`) |
| `bank-full.csv` | Bank Marketing | 45,211 | `marital` (married / single / divorced) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) — file `bank-full.csv` (`;`-separated) |

Missing values coded as `?` or `unknown` are treated as missing.
The data files are not redistributed in this repository; please respect the
licences of the original sources.
