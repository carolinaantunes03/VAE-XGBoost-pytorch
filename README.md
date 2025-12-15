git clone https://github.com/carolinaantunes03/VAE-XGBoost-pytorch.git

1. Dataset: 
https://drive.google.com/drive/folders/1AodzO73SRZ3JIHeVltzJEmgjzaXvDJAh?usp=sharing

(o ficheiro .gz é o dataset original e o ficheiro .txt é o que resulta do script brca_prepocessing.py)

2. Após fazer download do dataset correr o script VAE_models/Data_transformation_comparison_withVAE(6_layers_z50)(v1.0).py 

NOTA 1: mudar nome do ficheiro de output 
NOTA 2: foram feitos alguns comentários relativos a alterações do código comparado com a versão original em Tensorflow e também a outras alterações que fizemos para adapar o código consoante os nossos recursos computacionais

O ficheiro de outuput vai ficar guardado em counts_data/vae_compressed

3. Correr o script R_scripts_for_adding_labels/R_scripts_for_adding_labels(BRCA).R para selecionar as samples que têm label no espaço comprimido

4. Correr o script R_scripts_for_adding_labels/raw_counts_add_labels(BRCA).R para selecionar as samples que têm label na versão raw

NOTA 1: mudar nome do ficheiro de input e de output 

Os ficheiros vão ficar guardados em counts_data/vae_compressed_wLabels e counts_data/counts_data_with_label, respetivamente

5. Correr o script Benchmark_codes_xgboost/30_runs_xgboost.py para a pipeline VAE-XGBoost
6. Correr o script Benchmark_codes_xgboost/xgb_raw_data.py para a pipeline raw-XGBoost

NOTA 1: mudar nome do ficheiro de input 

7. Para a parte de interpretação correr primeiro o script Interpretation/latent_gene_correlation.py para calcular a correlação de pearson. De seguida correr id_to_gene_symbols.py para passar os EnsemblesIDs para Gene Symbols. Por fim correr run_gprofiler.py para obter os resultados finais de interpretação.

NOTA 1: mudar nomes dos ficheiros de input e output