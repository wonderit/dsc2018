library(ggplot2)   
library(gridExtra)  
library(reshape2)

data_ = read.csv("../data/speeddating.csv");
train = read.csv("../data/speeddating_train.csv");
test = read.csv("../data/speeddating_test.csv");

missing_column = apply(data, 2, function(v) sum(v=="?"))
write.csv(missing_column, file="../data/missing_column.csv");
