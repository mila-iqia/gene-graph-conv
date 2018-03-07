#group            name       otype  dclass     dim
#0     / expression_data H5I_DATASET   FLOAT   x 440
#1     /      gene_names H5I_DATASET  STRING    2001
#2     /      graph_data H5I_DATASET   FLOAT  x 2001
#3     /     labels_data H5I_DATASET INTEGER     440
#4     /    sample_names H5I_DATASET  STRING     525

write_hdf5_syn_data = function(sim_data, adj){
  file_name = "syn_nslr_1.hdf5"
  h5createFile(file_name)
  attr(sim_data$y, "0") <- "dead"
  attr(sim_data$y, "1") <- "survived"
  h5writeAttribute
  h5write(t(sim_data$X), file_name, "expression_data") 
  h5write(adj, file_name, "graph_data")
  names <- sprintf("name[%s]",seq(0:length(sim_data$w) - 1))
  h5write(names, file_name, "gene_names") 
  h5write(sim_data$y, file_name, "labels_data", write.attributes=TRUE) 
  labels <- sprintf("label[%s]",seq(0:length(sim_data$y) - 1))
  h5write(labels, file_name, "sample_names") 
  H5close()
}

write_hdf5_lung_data = function(sim_data, adj){
  file_name = "syn_nslr_1.hdf5"
  h5createFile(file_name)
  attr(sim_data$y, c("0", "1")) <- c("dead", "alive")
  
  h5write(t(sim_data$X), file_name, "expression_data") 
  h5write(adj, file_name, "graph_data")
  names <- sprintf("name[%s]",seq(0:length(sim_data$w) - 1))
  h5write(names, file_name, "gene_names") 
  h5write(sim_data$y, file_name, "labels_data", write.attributes=TRUE) 
  labels <- sprintf("label[%s]",seq(0:length(sim_data$y) - 1))
  h5write(labels, file_name, "sample_names") 
  H5close()
}

read_hdf5_data = function(){
  h5ls("/data/lisa/data/genomics/TCGA/syn_nslr.hdf5")
}