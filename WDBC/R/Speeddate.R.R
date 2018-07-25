library(ggplot2)   
library(glmnet)
library(randomForest)
library(neuralnet)
library(xgboost)
library(validann)
library(glasso)
library(igraph)
library(network)

data = read.csv("../data/speeddating.csv");
train = read.csv("../data/speeddating_train_preprocessed.csv");
test = read.csv("../data/speeddating_test_preprocessed.csv");

train.col = colnames(train)
yvar = "decision_o"
remove.var = yvar;
#remove.var = c("gender", "samerace", "met", "importance_same_race", yvar)
#               "pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny",
#               "pref_o_ambitious", "pref_o_shared_interests", yvar);

use.xvar = setdiff(train.col, remove.var)

xtrain = as.matrix(train[,use.xvar]);
ytrain = train[,yvar];
xtest = as.matrix(test[,use.xvar]);
ytest = test[,yvar];

# logistic lasso regression
cv.mod_lasso = cv.glmnet(x = xtrain, y=ytrain, alpha=1, family = "binomial")
lambda = cv.mod_lasso$lambda.1se;
mod_lasso = glmnet(x = xtrain, y=ytrain, lambda = lambda, family = "binomial");
pred_lasso = predict(mod_lasso, xtest, type="response")
(lasso.acc = mean(ytest==round(pred_lasso)))


# random forest
mod_randomf = randomForest(x = xtrain, y = as.factor(ytrain))
pred_randomf = predict(mod_randomf, xtest)
(randomf.acc = mean(ytest==pred_randomf))

# logistic regression
score_logis = rep(ncol(xtrain));
for(i in 1:ncol(xtrain)){
  tmp.data = data.frame(x=xtrain[,i], y=ytrain);
  mod_logis = glm(y~x, data=tmp.data, family = binomial(link = 'logit'))
  tmp.test = data.frame(x=xtest[,i]);
  tmp.pred_logis = predict(mod_logis, tmp.test, type = "response");
  score_logis[i]=mean(ytest==round(tmp.pred_logis))
}

# xgboost
mod_xgb = xgboost(data = xtrain, label = ytrain, nround=20, objective = "binary:logistic")
pred_xgb = predict(mod_xgb, xtest)
(xgb.acc = mean(ytest==(pred_xgb>0.5)))


# single layer ANN
n = names(train)
f = as.formula(paste("decision_o ~", paste(n[!n %in% "decision_o"], collapse = "+")))
nn = ann(x=train[,-31], y=train[,31], size=5, act_hid="sigmoid", act_out = "tanh")
pred.nn = predict(nn, test[,-31]);
(ann.acc = mean(ytest==(pred.nn>0.5)))


# SVM
#mod_svm = svm(f, data=train)
mod_svm = tune(svm, train.x=xtrain, train.y=ytrain, kernel="radial", ranges = list(cost=10^(-1:2), gamma=c(.5,1,2)))
pred.svm = predict(mod_svm$best.model, xtest)
(svm.acc = mean(test[,31]==(pred.svm>0.5)))



# glasso
xdata = rbind(xtrain,xtest);
n=nrow(xdata)
S.var = cov(xdata)
nr=1000
max.rho=max(abs(S.var[upper.tri(S.var)]))
rho = seq(0,max.rho, length=nr+1)[-1]
bic = rho
for(j in 1:nr){
  a       <- glasso(S.var,rho[j])
  p_off_d <- sum(a$wi!=0 & col(S.var)<row(S.var))
  bic[j]  <- -2*(a$loglik) + p_off_d*log(n)
}
best <- which.min(bic)
plot(rho,bic)
points(rho[best],bic[best],pch=19)

a = glasso(S.var, rho[best])
RES = a$wi
rownames(RES) = rownames(S.var)
colnames(RES) = colnames(S.var)
diag(RES)=0

network = NULL;
for(i in 1:nrow(RES)){
  for(j in i:ncol(RES)){
    if(RES[i,j]!=0){
      network = rbind(network, c(rownames(RES)[i],colnames(RES)[j],RES[i,j]))
    }
  }
}

g = graph(c(t(network[,1:2])), directed = FALSE)

#A = ifelse(RES!=0 & row(RES)!=col(RES), 1,0)
#g <- network(A, directed=FALSE)
tiff("../data/Correlation Network.tiff")
plot(g, vertex.label.dist=2, vertex.label.color="black", vertex.size=7, vertex.label.font=2)
dev.off()




