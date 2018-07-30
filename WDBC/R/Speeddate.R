library(ggplot2)   
library(glmnet)
library(randomForest)
library(neuralnet)
library(xgboost)
library(class)
library(validann)
library(glasso)
library(igraph)
library(network)
library(caret)
library(iterators)
library(rpart)
library(RWeka)
library(e1071)

for(stage in 1:3){
  nM = 12;
  rmse = data.frame(Origin=rep(0, nM),
                    Norm=rep(0, nM),
                    MinMax=rep(0, nM),
                    Quan=rep(0, nM),
                    Tran=rep(0, nM))
  rownames(rmse) = c("knn", "linear regression", "multi linear regression", "ridge",
                     "lasso", "elastic net", "bagging cart", "random forest",
                     "gradient boost", "svm", "single layer ann", "xgboost")
  R2 = rmse;
  
  R_sq = function(y, pred){
    ym = mean(y);
    sstot = sum((y-ym)^2);
    ssres = sum((y-pred)^2);
    r2 = 1 - (ssres/sstot);
  }
  
  
  #data = read.csv("../data/speeddating.csv");
  train = read.csv("../data/speeddating_likeo_train_yj.csv");
  train = train[,-(1:2)];
  train = train[,!(colnames(train)=="decision_o")]
  test = read.csv("../data/speeddating_likeo_test_yj.csv");
  test = test[,-(1:2)];
  test = test[,!(colnames(test)=="decision_o")]
  
  xtrain.quan = read.csv("../data/quantile_m_o_train_x.csv")
  xtest.quan = read.csv("../data/quantile_m_o_test_x.csv")
  
  xtrain.tran = read.csv("../data/power_m_o_train_x.csv")
  xtest.tran = read.csv("../data/power_m_o_test_x.csv")
  
  
  
  train.col = colnames(train)
  yvar = "like_o"

  if(stage==1){
    remove.var = c("dec_m", yvar);
  } else if(stage==2){
    remove.var = c("dec_m", "att_m", 
                   "sin_m", "int_m", "fun_m", "amb_m", "sha_m", "dec_m", yvar);
  } else if(stage==3){
    remove.var = c("dec_m", "attractive_o", "sinsere_o", "intelligence_o",
                   "funny_o", "ambitous_o", "shared_interests_o", "att_m", 
                   "sin_m", "int_m", "fun_m", "amb_m", "sha_m", "dec_m", yvar);
  }
  use.xvar = setdiff(train.col, remove.var);
  write.table(use.xvar, file = sprintf("../Res/stage%02d.txt", stage), col.names = FALSE, row.names = FALSE);
  
  # Original data
  xtrain =as.matrix(train[,use.xvar])
  ytrain = train[,yvar];
  xtest = as.matrix(test[,use.xvar]);
  ytest = test[,yvar];
  
  # Normalized data
  xtrain.N = scale(xtrain);
  xtest.N = scale(xtest);
  
  # MinMax scale
  MinMax = function(x){ return((x-min(x))/(max(x)-min(x))); }
  xtrain.MM = apply(xtrain, 2, MinMax);
  xtest.MM = apply(xtest, 2, MinMax);
  
  # Quantile scale
  xtrain.q = as.matrix(xtrain.quan[,use.xvar]);
  xtest.q = as.matrix(xtest.quan[,use.xvar]);
  
  # Power transform
  xtrain.t = as.matrix(xtrain.tran[,use.xvar]);
  xtest.t = as.matrix(xtest.tran[,use.xvar]);
  
  xtrain = list(O=xtrain, N=xtrain.N, MM=xtrain.MM, Q=xtrain.q, P=xtrain.t);
  xtest = list(O=xtest, N=xtest.N, mm=xtest.MM, Q=xtest.q, P=xtest.t);
  
  for(i in 1:ncol(rmse)){
    # full model formula
    n = c("like_o", colnames(xtrain[[i]]));
    f = as.formula(paste("like_o ~", paste(n[!n %in% "like_o"], collapse = "+")));
    tmp.train = data.frame(xtrain[[i]], "like_o"=ytrain);
    
    
    # knn regression
    mod_knn = knnreg(x=xtrain[[i]], y=ytrain, k=20)
    pred.knn = predict(mod_knn, xtest[[i]]);
    (rmse["knn", i] = sqrt(mean((ytest-pred.knn)^2)));
    R2["knn", i] = R_sq(ytest, pred.knn);
    print("KNN done");
    
    # linear regression via each variable
    score_lm = rep(0,ncol(xtrain[[i]]));
    r2_lm = rep(0,ncol(xtrain[[i]]));
    for(j in 1:ncol(xtrain[[i]])){
      tmp.data = data.frame(x=xtrain[[i]][,j], y=ytrain);
      mod_lm = lm(y~x, data=tmp.data)
      tmp.test = data.frame(x=xtest[[i]][,j]);
      tmp.pred_lm = predict(mod_lm, tmp.test);
      score_lm[j]= sqrt(mean((ytest-tmp.pred_lm)^2))
      r2_lm[j] = R_sq(ytest, tmp.pred_lm);
    }
    (rmse["linear regression", i] = min(score_lm));
    R2["linear regression", i] = max(r2_lm);
    print("Linear regression done")
    
    # linear regression via all variables
    mod_alm = lm(f, data=tmp.train)
    pred.alm = predict(mod_alm, as.data.frame(xtest[[i]]))
    (rmse["multi linear regression", i] = sqrt(mean((ytest-pred.alm)^2)))
    R2["multi linear regression", i] = R_sq(ytest, pred.alm);
    print("Mulit linear regression done")
    
    # ridge regression
    cv.mod_ridge = cv.glmnet(x = xtrain[[i]], y=ytrain, alpha=0)
    lambda = cv.mod_ridge$lambda.1se;
    mod_ridge = glmnet(x = xtrain[[i]], y=ytrain, lambda = lambda, alpha=0);
    pred.ridge = predict(mod_ridge, xtest[[i]])
    (rmse["ridge", i] = sqrt(mean((ytest-pred.ridge)^2)));
    R2["ridge", i] = R_sq(ytest, pred.ridge);
    print("Ridge done");
    
    # lasso regression
    cv.mod_lasso = cv.glmnet(x = xtrain[[i]], y=ytrain, alpha=1)
    lambda = cv.mod_lasso$lambda.1se;
    mod_lasso = glmnet(x = xtrain[[i]], y=ytrain, lambda = lambda, alpha=1);
    pred.lasso = predict(mod_lasso, xtest[[i]])
    (rmse["lasso", i] = sqrt(mean((ytest-pred.lasso)^2)));
    R2["lasso", i] = R_sq(ytest, pred.lasso);
    print("Lasso done");
    
    # elastic net
    cv.mod_els = cv.glmnet(x = xtrain[[i]], y=ytrain, alpha=0.5)
    lambda = cv.mod_els$lambda.1se;
    mod_els = glmnet(x = xtrain[[i]], y=ytrain, lambda = lambda, alpha=0.5);
    pred.els = predict(mod_els, xtest[[i]])
    (rmse["elastic net", i] = sqrt(mean((ytest-pred.els)^2)));
    R2["elastic net", i] = R_sq(ytest, pred.els);
    print("Elastic net done");
    
    # decision tree
    mod_dt = Bagging(f, data=tmp.train)
    pred.dt = predict(mod_dt, as.data.frame(xtest[[i]]))
    rmse["bagging cart", i] = sqrt(mean((ytest-pred.dt)^2))
    R2["bagging cart", i] = R_sq(ytest, pred.dt);
    print("Bagging CART done");
    
    # random forest
    mod_randomf = randomForest(x = xtrain[[i]], y = ytrain)
    pred.randomf = predict(mod_randomf, xtest[[i]])
    (rmse["random forest", i] = sqrt(mean((ytest-pred.randomf)^2)))
    R2["random forest", i] = R_sq(ytest, pred.randomf);
    print("Random forest done");
    
    # gradient boost
    caret.train.ctrl = trainControl(method="repeatedcv", number=5, repeats=5, verboseIter=FALSE, allowParallel=FALSE);
    mod_gbm = train(f, method="gbm", metric="RMSE", maximize=FALSE, trControl=caret.train.ctrl,
                    tuneGrid=expand.grid(n.trees=(4:10)*50, interaction.depth=c(5), shrinkage=c(0.05),
                                         n.minobsinnode=c(10)), data=tmp.train, verbose=FALSE);
    pred.gbm = predict(mod_gbm, xtest[[i]])
    rmse["gradient boost", i] = sqrt(mean((ytest-pred.gbm)^2))
    R2["gradient boost", i] = R_sq(ytest, pred.gbm);
    print("Gradient boost done")
    
    # SVM
    mod_svm = svm(f, data=tmp.train)
    #mod_svm = tune(svm, train.x=xtrain, train.y=ytrain, kernel="radial", ranges = list(cost=10^(-1:2), gamma=c(.5,1,2)))
    pred.svm = predict(mod_svm, xtest[[i]])
    (rmse["svm", i] = sqrt(mean((test[,"like_o"]-pred.svm)^2)))
    R2["svm", i] = R_sq(ytest, pred.svm);
    print("SVM done");
    
    # single layer ANN
    nn = ann(x=xtrain[[i]], y=ytrain, size=5, act_hid="sigmoid", act_out = "linear")
    pred.nn = predict(nn, xtest[[i]]);
    (rmse["single layer ann", i] = sqrt(mean((ytest-pred.nn)^2)))
    R2["single layer ann", i] = R_sq(ytest, pred.nn);
    print("ANN done");
    
    # xgboost
    mod_xgb = xgboost(data = xtrain[[i]], label = ytrain, nround=1000, verbose = FALSE, 
                      objective="reg:linear", eval_metric="rmse", eta=0.01, gamma=0.05, max_depth=6,
                      min_child_weight = 1.7817, subsample = 0.5213, colsample_bytree = 0.4603)
    pred.xgb = predict(mod_xgb, xtest[[i]])
    (rmse["xgboost", i] = sqrt(mean((ytest-pred.xgb)^2)))
    R2["xgboost", i] = R_sq(ytest, pred.xgb);
    print("Xgboost done");
  }
  write.table(rmse, col.names = TRUE, row.names = TRUE, file=sprintf("../Res/RMSE_stage%02d.txt",stage))
  write.table(R2, col.names = TRUE, row.names = TRUE, file=sprintf("../Res/Rsquare_stage%02d.txt",stage))
  
  # multi-layer ANN
  #mnn = neuralnet(f, data=train, hidden=c(5,3), linear.output=TRUE)
  #pred.mnn = compute(mnn, xtest);
  #(mnn.acc = sqrt(mean((ytest-pred.mnn$net.result)^2)))
  
}










