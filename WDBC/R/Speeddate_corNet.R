library(glasso)

# Correlation Network by graphical lasso
train = read.csv("../data/speeddating_likeo_train_yj.csv");
test = read.csv("../data/speeddating_likeo_test_yj.csv");

xdata = rbind(train[,c(-1,-2)],test[,c(-1,-2)]);
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
tiff("../Res/Cross-validation_GLasso.tiff", width=1440, height = 1440)
par(mar=c(10,10,8,2), mgp=c(6,2,0))
plot(log(rho),bic, type='l', xlab = "log(Lambda)", ylab = "BIC", lwd=4, cex=4, cex.main=4, cex.lab=4, cex.axis=3, main="Cross-Validation for Graphical Lasso")
points(log(rho[best]),bic[best],pch=19, col="red", cex=3)
legend("bottomright", bty='n', legend = "Lambda.max", pch=19, col="red", cex=4)
dev.off();

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
tiff("../Res/Correlation Network.tiff", width = 1440, height = 1440)
par(mar=c(5,5,8,5))
plot(g, vertex.label.dist=1.5, vertex.label.color="black", vertex.size=5, vertex.label.font=2, vertex.label.cex=2, cex.main=4, vertex.label.degree=-pi/2)
dev.off();


