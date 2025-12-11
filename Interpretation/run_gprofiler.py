from gprofiler import GProfiler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURATION ===
gene_file = "gene_symbols_above_0.4_z45.txt"
output_file = "gprofiler_results__above_0.4z45.tsv"

# === LOAD GENE LIST ===
with open(gene_file) as f:
    gene_list = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(gene_list)} genes for enrichment analysis")

# === RUN g:Profiler ===
gp = GProfiler(return_dataframe=True)
results = gp.profile(
    organism="hsapiens",
    query=gene_list,
    sources=["GO:BP", "KEGG", "REAC"],
)

# === SAVE RESULTS ===
results.to_csv(output_file, sep="\t", index=False)
print(f"\n✅ Saved enrichment results: {output_file}")

# === DISPLAY TOP RESULTS ===
# Check which intersection column exists
intersection_col = None
for col in ["intersections", "intersection", "intersect", "intersection_genes"]:
    if col in results.columns:
        intersection_col = col
        break

# Choose which columns to display safely
display_cols = ["source", "name", "p_value", "term_size", "intersection_size"]
if intersection_col:
    display_cols.append(intersection_col)

print("\nTop 10 enriched terms:")
print(results[display_cols].head(10).to_string(index=False))

# === PLOT TOP RESULTS ===
top = results.sort_values("p_value").head(10)

plt.figure(figsize=(8,5))
sns.barplot(
    data=top,
    x=-np.log10(top["p_value"]),
    y=top["name"],
    hue="source",
    dodge=False
)
plt.xlabel("-log10(p-value)")
plt.ylabel("Pathway / Process")
plt.title("Top 10 Enriched Biological Processes (g:Profiler)")
plt.tight_layout()
plt.savefig("gprofiler_top10_z45.png", dpi=300)
plt.show()
