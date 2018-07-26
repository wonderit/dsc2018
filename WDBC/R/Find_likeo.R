# Find like_o
data = read.csv("../data/speeddating_id.csv")
train = read.csv("../data/speeddating_preprocessed_id_mean_train.csv");
test = read.csv("../data/speeddating_preprocessed_id_mean_test.csv");
train = data.frame(train, like_o = rep(-1,nrow(train)))
test = data.frame(test, like_o = rep(-1, nrow(test)))

# train
for(i in 1:nrow(train)){
  idx = train[,"iid"]==train[i,"pid"]&train[,"pid"]==train[i,"iid"];
  train[idx,"like_o"] = train[i,"like"];
}

#test
for(i in 1:nrow(test)){
  idx = test[,"iid"]==test[i,"pid"]&test[,"pid"]==test[i,"iid"];
  test[idx,"like_o"] = test[i,"like"];
}

# fill NA in train
NA.idx = which(train[,"like_o"]<0);
for(i in NA.idx){
  idx = data[,"iid"]==train[i,"pid"]&data[,"pid"]==train[i,"iid"];
  if(as.character(data[idx,"like"])=="?") {next;}
  train[i,"like_o"] = as.numeric(as.character(data[idx,"like"]));
}

# remove NA of like_o in train
NA.idx = which(train[,"like_o"]<0);
train = train[-NA.idx,]

# fill NA in test
NA.idx = which(test[,"like_o"]<0);
for(i in NA.idx){
  idx = data[,"iid"]==test[i,"pid"]&data[,"pid"]==test[i,"iid"];
  if(as.character(data[idx,"like"])=="?") {next;}
  test[i,"like_o"] = as.numeric(as.character(data[idx,"like"]));
}

# remove NA of like_o in test
NA.idx = which(test[,"like_o"]<0);
test = test[-NA.idx,]

write.csv(train, file="../data/speeddating_likeo_train_yj.csv", row.names = FALSE)
write.csv(test, file="../data/speeddating_likeo_test_yj.csv", row.names = FALSE)



