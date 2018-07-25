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
  print(i);
}

#test
for(i in 1:nrow(test)){
  idx = test[,"iid"]==test[i,"pid"]&test[,"pid"]==test[i,"iid"];
  test[idx,"like_o"] = test[i,"like"];
  print(i);
}

# fill NA in train
NA.idx = which(train[,"like_o"]<0);
for(i in NA.idx){
  idx = data[,"iid"]==train[i,"pid"]&data[,"pid"]==train[i,"iid"];
  train[i,"like_o"] = data[idx,"like"];
  print(i);
}

# fill NA in test
NA.idx = which(test[,"like_o"]<0);
for(i in NA.idx){
  idx = data[,"iid"]==test[i,"pid"]&data[,"pid"]==test[i,"iid"];
  test[i,"like_o"] = data[idx,"like"];
  print(i);
}

write.csv(train, file="../data/speeddating_preprocessed_id_mean_likeo_train.csv")
write.csv(test, file="../data/speeddating_preprocessed_id_mean_likeo_test.csv")


