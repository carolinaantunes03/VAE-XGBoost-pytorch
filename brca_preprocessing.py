# prepress_brca_expression.py
import pandas as pd
import numpy as np

# === 1. Load UCSC Xena BRCA expression file ===
# Example file: "TCGA-BRCA.htseq_fpkm.tsv"
# It should have rows = genes, columns = samples
path = "TCGA-BRCA.star_fpkm.tsv.gz"
expr = pd.read_csv(path, sep="\t", index_col=0)

# === 2. Transpose so samples are rows ===
expr = expr.T

# === 3. Drop non-numeric and constant columns ===
expr = expr.loc[:, expr.var() > 0]

# === 4. Handle missing values (if any) ===
expr = expr.dropna(axis=1)

# === 5. Optional: min-max normalization (0–1) ===
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
expr_scaled = pd.DataFrame(
    scaler.fit_transform(expr),
    index=expr.index,
    columns=expr.columns
)

# === 6. Save preprocessed data ===
expr_scaled.to_csv(
    "TCGA_BRCA_VSTnorm_count_expr_clinical_data.txt",
    sep="\t"
)

print("Saved preprocessed BRCA expression data:", expr_scaled.shape)
