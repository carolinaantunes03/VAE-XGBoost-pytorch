import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# === CONFIGURATION ===
latent_file = "../counts_data/vae_compressed/encoded_BRCA_VAE_z50_pytorch_exp3.tsv"
expression_file = "../counts_data/counts_data_with_label/TCGA_BRCA_VSTnorm_count_expr_clinical_data.txt"
latent_feature = "45"   # f44 corresponds to latent variable X45
output_file = f"gene_correlation_z{latent_feature}.tsv"



# === LOAD DATA ===
print("Loading data...")
latent_df = pd.read_csv(latent_file, sep="\t", index_col=0)

# For the expression file, use Ensembl_ID as the sample index
expr_df = pd.read_csv(expression_file, sep="\t")
expr_df = expr_df.set_index("Ensembl_ID")

# Drop any non-gene metadata columns (like response_group)
if "response_group" in expr_df.columns:
    expr_df = expr_df.drop(columns=["response_group"])


# === ALIGN SAMPLES ===
# Clean sample names just in case
latent_df.index = latent_df.index.str.strip()
expr_df.index = expr_df.index.str.strip()

# Find intersection
common_samples = latent_df.index.intersection(expr_df.index)
latent_df = latent_df.loc[common_samples]
expr_df = expr_df.loc[common_samples]

print("Number of latent samples:", len(latent_df))
print("Number of expression samples:", len(expr_df))
print("Number of common samples:", len(common_samples))

print(f"Using {len(common_samples)} common samples.")
print(f"Expression matrix shape: {expr_df.shape}")

# === SELECT LATENT FEATURE VECTOR ===
z_vector = latent_df[f"{latent_feature}"].astype(float)
print(f"Selected latent feature z{latent_feature} with mean={z_vector.mean():.4f}, std={z_vector.std():.4f}")

# === OPTIONAL: REMOVE ZERO-VARIANCE GENES ===
expr_df = expr_df.loc[:, expr_df.var(axis=0) > 0]

# === VECTORIZE CORRELATION COMPUTATION ===
print("Calculating Pearson correlations (vectorized)...")

# Standardize both z_vector and expression data
z_std = (z_vector - z_vector.mean()) / z_vector.std(ddof=0)
expr_std = (expr_df - expr_df.mean(axis=0)) / expr_df.std(axis=0, ddof=0)

# Compute Pearson correlation: r = (Xᵀz) / (n - 1)
n = len(z_std)
r_values = expr_std.T.dot(z_std) / (n - 1)

# Convert to DataFrame
cor_df = pd.DataFrame({
    "Gene": r_values.index,
    "Pearson_r": r_values.values
})
cor_df["abs_r"] = cor_df["Pearson_r"].abs()

# Sort by absolute correlation
cor_df = cor_df.sort_values(by="abs_r", ascending=False)

# === SAVE RESULTS ===
cor_df.to_csv(output_file, sep="\t", index=False)
print(f"\n✅ Saved correlation table: {output_file}")

# === PRINT SUMMARY ===
print("\nTop 10 most correlated genes with z" + latent_feature + ":")
print(cor_df.head(10))

# === SELECT TOP 500 GENES ===
top_genes = cor_df.head(1000)["Gene"].tolist()
top_output_file = f"top1000_gene_correlation_z{latent_feature}.tsv"
cor_df.head(1000).to_csv(top_output_file, sep="\t", index=False)
print(f"\n✅ Saved top 1000 correlated genes: {top_output_file}")


# SELECT GENES WITH CORRELATION >= 0.2

threshold = 0.4
filtered_df = cor_df[cor_df["abs_r"] >= threshold]
filtered_output_file = f"gene_correlation_z{latent_feature}_absr_above_{threshold}.tsv"
filtered_df.to_csv(filtered_output_file, sep="\t", index=False)
print(f"\n✅ Saved genes with |Pearson_r| >= {threshold}: {filtered_output_file}")

import matplotlib.pyplot as plt
import seaborn as sns

# === VISUALIZE CORRELATION DISTRIBUTION ===
plt.figure(figsize=(8,5))
sns.histplot(cor_df["Pearson_r"], bins=100, color="#2E86AB", kde=True)
plt.title(f"Distribution of Pearson correlations with latent feature z{latent_feature}")
plt.xlabel("Pearson correlation (r)")
plt.ylabel("Number of genes")
plt.axvline(x=0.6, color="red", linestyle="--", label="|r| = 0.6")
plt.axvline(x=-0.6, color="red", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig(f"correlation_distribution_z{latent_feature}.png", dpi=300)
plt.show()

top_n = 20
plt.figure(figsize=(6,4))
sns.barplot(
    data=cor_df.head(top_n),
    x="abs_r", y="Gene", color="#1F77B4"
)
plt.title(f"Top {top_n} genes correlated with latent feature z{latent_feature}")
plt.xlabel("|Pearson r|")
plt.ylabel("Gene ID")
plt.tight_layout()
plt.savefig(f"top{top_n}_correlated_genes_z{latent_feature}.png", dpi=300)
plt.show()
