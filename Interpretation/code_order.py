import pandas as pd

# 1. Ler o ficheiro TSV
filename = 'Interpretation/gene_correlation_z45_absr_above_0.2.tsv'
df = pd.read_csv(filename, sep='\t')

# 2. Criar/Garantir a coluna de Valor Absoluto
# Isto transforma valores como -0.8 em 0.8 para efeitos de ordenação
df['abs_r'] = df['Pearson_r'].abs()

# 3. Limpar o ID do gene (remover as versões .8, .9, etc.)
# O g:Profiler prefere "ENSG00000245849" a "ENSG00000245849.8"
df['Gene_clean'] = df['Gene'].astype(str).str.split('.').str[0]

# 4. ORDENAR (Passo Crucial)
# Ordena do maior valor absoluto para o menor
df_sorted = df.sort_values(by='abs_r', ascending=False)

# 5. Selecionar apenas a lista de IDs
gene_ids = df_sorted['Gene_clean']

# 6. Exportar para .txt
output_filename = 'genes_ordenados_absoluto.txt'
gene_ids.to_csv(output_filename, index=False, header=False)

print(f"Sucesso! Ficheiro '{output_filename}' criado.")