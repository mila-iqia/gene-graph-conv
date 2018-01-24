# -------------------------------------------------------------------------
get.penalityMatrix=function(adj,X1, y1){
  feature.num = dim(adj)[2] 
  M.c = diag(0,feature.num)
  M.lasso = diag(0,feature.num)
  M.elastic = diag(1,feature.num)  
  M.network = Non.NormalizedLaplacianMatrix(adj)
  M.AdaptNet = AdaptNet.Non.NormalizedLap(adj,X1, y1)
  return(list(M.c=M.c,M.lasso=M.lasso,M.elastic=M.elastic,M.network=M.network,M.AdaptNet=M.AdaptNet))
}
# -------------------------------------------------------------------------
# Normalized Laplacian Matrix from adjacency matrix
laplacianMatrix = function(adj){
  diag(adj) <- 0
  deg <- apply(abs(adj),1,sum)
  p <- ncol(adj)
  L <- matrix(0,p,p)
  nonzero <- which(deg!=0)
  for (i in nonzero){
    for (j in nonzero){
      L[i,j] <- -adj[i,j]/sqrt(deg[i]*deg[j])
    }
  }
  diag(L) <- 1
  return(L)
}

# Non-Normalized Laplacian Matrix from adjacency matrix
Non.NormalizedLaplacianMatrix = function(adj){
  diag(adj) <- 0
  deg <- apply(adj,1,sum)
  D = diag(deg)
  L = D - adj
  return(L)
}
# ------------------------------------------------------
AdaptNet.penality.matrix = function(adj, X, y){
  
  library(glmnet)
  glmnet.fit = glmnet(X, y, lambda=0, family='binomial')
  Beta = coef(glmnet.fit)
  
  p <- ncol(adj)
  coeff.sign = sign(Beta)[2:(p+1)]
  
  diag(adj) <- 0
  deg <- apply(abs(adj),1,sum)
  L <- matrix(0,p,p)
  nonzero <- which(deg!=0)
  for (i in nonzero){
    for (j in nonzero){
      temp.sign = coeff.sign[i]*coeff.sign[j]
      L[i,j] <- -temp.sign*adj[i,j]/sqrt(deg[i]*deg[j])
    }
  }
  diag(L) <- 1
  return(L_star=L)
}
AdaptNet.Non.NormalizedLap = function(adj,X, y){
  library(glmnet)
  glmnet.fit = glmnet(X, y, lambda=0, family='binomial')
  Beta = coef(glmnet.fit)
  
  p <- ncol(adj)
  coeff.sign = sign(Beta)[2:(p+1)]
  diag(adj) <- 0
  deg <- apply(abs(adj),1,sum)
  L <- matrix(0,p,p)
  nonzero <- which(deg!=0)
  for (i in nonzero){
    for (j in nonzero){
      temp.sign = coeff.sign[i]*coeff.sign[j]
      L[i,j] <- -temp.sign*adj[i,j]
    }
  }
  diag(L) <- deg
  return(L_star=L)
}
# -------------------------------------------------------------------------
sen.spe = function(pred, truth){
  #----------------------------------------
  # 1: True sample
  # 0: False sample
  # truth = abs(sign(sim.data$w))
  # pred = abs(sign(out1$w[-length(tmp)]))
  #----------------------------------------
  sen = length(which(pred[which(truth==1)]==1))/length(which(truth==1))
  spe = length(which(pred[which(truth==0)]==0))/length(which(truth==0))
  result = c(sen,spe);names(result) = c("sensitivity","specificity")
  return(result)
}

# Prior network regularization
Prior.network = function(adj){
  # adj is a p x p matrix
  # L   is a (p+1) x (p+1) matrix
  L = laplacianMatrix(adj)
  L = rbind(L,rep(0,p))
  L = cbind(L,rep(0,p+1))
  return(L)
}
# -------------------------------------------------------------------------
get_sim_prior_Net = function(n,t,p11,p12){
  A = matrix(0,nrow=n,ncol=n) 
  for(i in 1:n){
    for(j in 1:n){
      if(i>j){
        set.seed(10*i+8*j)
        if(i<t&j<t){
          if(runif(1)<p11) A[i,j] = 1}
        else{
          if(runif(1)<p12) A[i,j] = 1}  
      }
    }
  }
  A = A +t(A)
  diag(A)=0
  return(A)
}
# -------------------------------------------------------------------------
GET.SIM.DATA = function(smaple.num, feature.num, random.seed, snrlam=0.05){
  # smaple.num = 700; feature.num = 100; random.seed = 10; snrlam=0.05
  ii = random.seed
  set.seed(30)
  w  <-c(rnorm(40),rep(0,(feature.num-40)));b = 0
  mu <- rep(0,40)
  Sigma <- matrix(.6, nrow=40, ncol=40) + diag(40)*.4
  
  set.seed(ii*2)
  X1 <- mvrnorm(n=smaple.num, mu=mu, Sigma=Sigma)
  
  set.seed(ii*3)
  X2 <- matrix(rnorm(smaple.num*(feature.num-40), mean = 0, sd = 1), nrow = smaple.num, ncol = feature.num-40)
  X = cbind(X1,X2)

  Xw <- -X%*%w 
  pp <- 1/(1+exp(Xw))
  y  <- rep(1,smaple.num)
  y[pp<0.5] <- 0

  p = dim(X)[1];q = dim(X)[2]
  set.seed(ii*3)
  XX = X + snrlam*matrix(rnorm(p*q),ncol=q)
  
  return(list(X=XX,y=y,w=w))
}
# -------------------------------------------------------------------------
GET.SIM.DATA2 = function(smaple.num, feature.num, random.seed, snrlam=0.05){
  # smaple.num = 700; feature.num = 100; random.seed = 10; snrlam=0.05
  ii = 10
  set.seed(30)
  w  <-c(rnorm(40),rep(0,(feature.num-40)));b = 0
  mu <- rep(0,40)
  Sigma <- matrix(.6, nrow=40, ncol=40) + diag(40)*.4
  
  set.seed(ii*2)
  X1 <- mvrnorm(n=smaple.num, mu=mu, Sigma=Sigma)
  
  set.seed(ii*3)
  X2 <- matrix(rnorm(smaple.num*(feature.num-40), mean = 0, sd = 1), nrow = smaple.num, ncol = feature.num-40)
  X = cbind(X1,X2)

  Xw <- -X%*%w 
  pp <- 1/(1+exp(Xw))
  y  <- rep(1,smaple.num)
  y[pp<0.5] <- 0
  
  p = dim(X)[1];q = dim(X)[2]
  
  set.seed(random.seed)
  XX = X + snrlam*matrix(rnorm(p*q),ncol=q)
  
  return(list(X=XX,y=y,w=w))
}