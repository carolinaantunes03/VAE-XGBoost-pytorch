import mygene
import pandas as pd

latent_feature = "45"   # f44 corresponds to latent variable X45

# Load your top 500 genes file
top_df = pd.read_csv(f"top500_gene_correlation_z{latent_feature}.tsv", sep="\t")

# Clean Ensembl IDs (remove version numbers like ".16")
top_df["Gene_clean"] = top_df["Gene"].str.split(".").str[0]

# Query mygene for gene symbols
mg = mygene.MyGeneInfo()
query_result = mg.querymany(
    top_df["Gene_clean"].tolist(),
    scopes="ensembl.gene",
    fields="symbol,name",
    species="human"
)

# Convert results to DataFrame and merge
mapping_df = pd.DataFrame(query_result)[["query", "symbol", "name"]].drop_duplicates("query")
top_df = top_df.merge(mapping_df, left_on="Gene_clean", right_on="query", how="left")

# Save mapped results
mapped_output_file = f"top500_gene_correlation_z{latent_feature}_mapped.tsv"
top_df.to_csv(mapped_output_file, sep="\t", index=False)
print(f"\n✅ Saved Ensembl → Symbol mapped file: {mapped_output_file}")
print(top_df.head(10))

# Load the mapped file
mapped_file = f"top500_gene_correlation_z{latent_feature}_mapped.tsv"
df = pd.read_csv(mapped_file, sep="\t")

# Keep only genes that were successfully mapped
rnk_df = df[["symbol", "Pearson_r"]].dropna()

# Drop duplicates (some Ensembl IDs map to the same symbol)
rnk_df = rnk_df.drop_duplicates(subset="symbol")

# Sort by correlation (descending for positive enrichment)
rnk_df = rnk_df.sort_values(by="Pearson_r", ascending=False)

# Save to GSEA format
rnk_file = f"gene_correlation_z{latent_feature}.rnk"
rnk_df.to_csv(rnk_file, sep="\t", header=False, index=False)
print(f"✅ GSEA rank file created: {rnk_file}")
print(rnk_df.head())