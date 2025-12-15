import pandas as pd
import mygene


input_file = "gene_correlation_z45_absr_above_0.4.tsv"
output_symbols_txt = "gene_symbols_above_0.4_z45.txt"
output_mapping_tsv = "ensembl_to_symbol_mapping__above_0.4z45.tsv"


df = pd.read_csv(input_file, sep="\t")
print(f"Loaded {len(df)} entries from {input_file}")

# Keep only Gene column and remove version numbers
df["Ensembl_ID"] = df["Gene"].str.split(".").str[0]
df = df[["Ensembl_ID"]].drop_duplicates()

# CONVERT TO GENE SYMBOLS
print("Converting Ensembl IDs to gene symbols via MyGene.info...")
mg = mygene.MyGeneInfo()
results = mg.querymany(df["Ensembl_ID"].tolist(), scopes="ensembl.gene", fields="symbol", species="human")

# Extract mapping
mapping = []
for r in results:
    ensembl_id = r["query"]
    symbol = r.get("symbol")
    mapping.append((ensembl_id, symbol if symbol is not None else "NA"))

mapping_df = pd.DataFrame(mapping, columns=["Ensembl_ID", "Gene_Symbol"])

# Merge back to preserve original order
merged_df = df.merge(mapping_df, on="Ensembl_ID", how="left")

merged_df.to_csv(output_mapping_tsv, sep="\t", index=False)
merged_df["Gene_Symbol"].dropna().to_csv(output_symbols_txt, index=False, header=False)

print(f"\nSaved mapping table: {output_mapping_tsv}")
print(f"Saved gene symbol list for g:Profiler: {output_symbols_txt}")
print(f"Total mapped symbols: {merged_df['Gene_Symbol'].notna().sum()} / {len(merged_df)}")
