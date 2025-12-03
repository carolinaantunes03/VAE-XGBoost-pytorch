import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# === CONFIGURATION ===
latent_file = "../VAE_models/counts_data/vae_compressed/encoded_BRCA_VAE_z50_pytorch_exp2.tsv"
expression_file = "../TCGA_BRCA_VSTnorm_count_expr_clinical_data.txt"
latent_feature = "45"   # Example: '1' for z1, '19' for z19
output_file = f"gene_correlation_z{latent_feature}.tsv"

# === LOAD DATA ===
print("Loading data...")
latent_df = pd.read_csv(latent_file, sep="\t", index_col=0)
expr_df = pd.read_csv(expression_file, sep="\t", index_col=0)

# Ensure samples match
common_samples = latent_df.index.intersection(expr_df.index)
latent_df = latent_df.loc[common_samples]
expr_df = expr_df.loc[common_samples]

print(f"Using {len(common_samples)} common samples.")

# === SELECT LATENT FEATURE ===
z_vector = latent_df[f"{latent_feature}"]  # column name = number (1..50)
z_vector = z_vector.astype(float)

# === CORRELATION LOOP ===
print(f"Calculating Pearson correlation between z{latent_feature} and {expr_df.shape[1]} genes...")

results = []
for gene in expr_df.columns:
    gene_expr = expr_df[gene].astype(float)
    if gene_expr.var() == 0:
        continue  # skip constant genes
    r, p = pearsonr(z_vector, gene_expr)
    results.append((gene, r, p))

cor_df = pd.DataFrame(results, columns=["Gene", "Pearson_r", "p_value"])
cor_df["abs_r"] = cor_df["Pearson_r"].abs()

# Sort by absolute correlation strength
cor_df = cor_df.sort_values(by="abs_r", ascending=False)

# === SAVE RESULTS ===
cor_df.to_csv(output_file, sep="\t", index=False)
print(f"Saved correlation table: {output_file}")

# === OPTIONAL: Print top genes ===
print("\nTop 10 genes most correlated with z" + latent_feature + ":")
print(cor_df.head(10))
