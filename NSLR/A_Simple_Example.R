library(pROC)
library(MASS)
library(glmnet)
# ------------------------------------------
# Please first set the path
# working_dir = "C:/Users/dell/Desktop/NSLR"
# setwd(working_dir)

source('Fun_Auxiliary.R')
source('Fun_NSLR.R')

# ------------------------------------------
# Generate a simple simulation data
snrlam0 = 3
f.num0  = 100
sim.data = GET.SIM.DATA2(smaple.num = 700, feature.num=f.num0, random.seed = 10, snrlam=snrlam0)
adj = get_sim_prior_Net(f.num0, 40, 0.3,0.05)

Train.id = 1:300
Valid.id = 301:500
w.true = sim.data$w

# Training data
X1 = sim.data$X[Train.id,]; y1 =sim.data$y[Train.id]; 
# Testing data
X2 = sim.data$X[Valid.id,]; y2 =sim.data$y[Valid.id]; 

# Non Normalized Laplacian Matrix"
PM = get.penalityMatrix(adj,X1, y1)

# ----------------------------------------
# ----------------------------------------
# Two regularization parameters
lambda = 0.2 
alpha  = 0.3

# ----------------------------------------
# A typical logistic regression model and four regularized logistic regession models
# ----------------------------------------

# Typical Logistic Regression Model
out1 =    SGNLR(X1, y1, PM$M.c,        lambda=0, alpha=0, niter=20)

# L1 (Lasso) Regularized Logistic Regression Model
out2 =    SGNLR(X1, y1, PM$M.lasso,    lambda,   alpha,   niter=20)

# Elastic Net Regularized Logistic Regression Model
out3 =    SGNLR(X1, y1, PM$M.elastic,  lambda,   alpha,   niter=20)

# Classical Network-regularized Logistic Regression Model
out4 =    SGNLR(X1, y1, PM$M.network,  lambda,   alpha,   niter=20)

# Adaptive Network-regularized Logistic Regression Model
out5 =    SGNLR(X1, y1, PM$M.AdaptNet, lambda,   alpha,   niter=20)

# Absolute Network-regularized Logistic Regression Model
out6 = abs.SNLR(X1, y1, PM$M.network,  lambda,   alpha,   niter=20)

# Testing
res1 = predict.SGNLR(X2,y2,out1$w)
res2 = predict.SGNLR(X2,y2,out2$w)
res3 = predict.SGNLR(X2,y2,out3$w)
res4 = predict.SGNLR(X2,y2,out4$w)
res5 = predict.SGNLR(X2,y2,out5$w)
res6 = predict.SGNLR(X2,y2,out6$w)

multi.method.AUC = c(res1$AUC,res2$AUC,res3$AUC,res4$AUC,res5$AUC,res6$AUC)
names(multi.method.AUC) = c("LR","Lasso.LR","Elastic.LR","Network.LR","AdaNet.LR","AbsNet.LR")

# Result in term of AUC
print(multi.method.AUC)










