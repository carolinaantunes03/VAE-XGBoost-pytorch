###########################################
#read in vae compressed data
#################################################
#expression data

#expr.data <- read.table("../VAE_models/counts_data/vae_compressed/encoded_BRCA_VAE_z50_pytorch_exp3.tsv"
expr.data <- read.table("counts_data/vae_compressed/encoded_BRCA_VAE_z50_pytorch_exp3.tsv", 
                        sep="\t",
                        header=TRUE, 
                        stringsAsFactors=FALSE, 
                        row.names=1,
                        quote="",
                        comment.char="#")
#################################
#clinical data
#cli.data <- read.table ("../binary_labels/TCGA-BRCA-binary-labels.txt", sep="\t",
cli.data <- read.table ("binary_labels/TCGA-BRCA-binary-labels.txt", sep="\t",
                        header=T, stringsAsFactors=FALSE,
                        quote="",
                        comment.char="#")
#colnames(cli.data) <- cli.data[1,]
cli.data <- cli.data[,-1]

#select only therapy_type and measure of response
colnames(cli.data)


#######################################
#merged dataframe
#expression data dim 1:60484
expr.temp.data <- expr.data #data.frame(t(expr.temp2.data))

expr.temp.data$Ensembl_ID <- rownames(expr.temp.data)
#expr.temp.data[,60484]

merge.data <- merge(expr.temp.data, cli.data, by.x = "Ensembl_ID", by.y = "submitter_id.samples", all.x = T)
colnames(merge.data)
#which(merge.data$therapy_type == "Chemotherapy")

##if using only labeled tumor samples then don not need to do the subsetting
#merge.chemo.full.data <- merge.chemo.data

#double check the # of positive/negative response
nrow(merge.data[which(merge.data$response_group == 1), ])
nrow(merge.data[which(merge.data$response_group == 0), ])



merge.chemo.label.data <- merge.data[-which(is.na(merge.data$response_group)),]

#merge.response.data <- merge.chemo.full.data[,c(1:(ncol(merge.chemo.full.data) - 3), ncol(merge.chemo.full.data))]

colnames(merge.chemo.label.data[,(ncol(merge.chemo.label.data) - 6):ncol(merge.chemo.label.data)])

#########################################################
#output file
#################################################################

#output_dir <- "../VAE_models/counts_data/vae_compressed_wLabels/"
output_dir <- "counts_data/vae_compressed_wLabels/"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.table(merge.chemo.label.data,
            file=paste0(output_dir, "encoded_BRCA_VAE_z50_withLabels_pytorch_exp3.txt"),
            sep="\t",
            quote=FALSE,
            row.names=T,
            col.names=NA)



####################################
#clean
#######################
#rm(list=ls(all=TRUE))
