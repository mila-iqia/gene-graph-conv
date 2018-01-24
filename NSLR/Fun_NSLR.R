update.diagMatrix = function(X, w, gate = 0){
  # ------------------
  # X   is a n x (p+1) matrix
  # w   is a (p+1) x 1 vector
  # ------------------
  n = nrow(X)
  # ------------------
  if(gate == 1){
    D = diag(n)*0.25
    return(D)
  }
  # ------------------
  pp = 1/(1 + exp(-X%*%w))
  D = diag(n)
  for(i in 1:n){
    D[i,i] = max(0.00001,pp[i]*(1-pp[i]))
    #D[i,i] = pp[i]*(1-pp[i])
  }
  return(D)
}
update.zVector= function(X, y, w, D){
  # ------------------
  # X   is a n x (p+1) matrix
  # y   is a n x 1 vector
  # w   is a (p+1) x 1 vector
  # D   is a n x n matrix
  # ------------------
  n = nrow(X)
  z = matrix(0,nrow  = n)
  
  Xw = X%*%w
  pp = 1/(1 + exp(-Xw))
  for(i in 1:n){
    z[i] = Xw[i] + (y[i]-pp[i])/D[i,i]
  }
  return(z)
}
# Soft-thresholding function
Soft.thresholding = function(x,lamb){
  if (lamb==0) {return(x)}
  value1=abs(x)-lamb; value1[round(value1,7)<=0]=0
  value2=sign(x)*value1
  return(value2)
}

# Objective function
objective = function(X,y,w){
  # -----------------
  # X   is a n x (p+1) matrix
  # y   is a n x 1 vector
  # w   is a (p+1) x 1 vector
  # ------------------
  value = t(y)%*%X%*%w - sum(log(1+exp(X%*%w)))
  value = value/length(y)
}

logreg.obj = function(X,y,w,M,lambda0, alpha){
  # -----------------
  # X   is a n x (p+1) matrix
  # y   is a n x 1 vector
  # w   is a (p+1) x 1 vector
  # ------------------
  s = X%*%w
  v = y * s - log(1+exp(s))
  L = mean(v)
  
  w1 = w[1:(length(w)-1)]
  R1 = lambda0 *alpha* sum(abs(w1))
  R2 = lambda0 *(1-alpha)*t(w)%*%M%*%w/2 
  R1 + R2 - L
}

abs.logreg.obj = function(X,y,w,M,lambda0, alpha){
  # -----------------
  # X   is a n x (p+1) matrix
  # y   is a n x 1 vector
  # w   is a (p+1) x 1 vector
  # ------------------
  s = X%*%w
  v = y * s - log(1+exp(s))
  L = mean(v)
  
  w1 = w[1:(length(w)-1)]
  R1 = lambda0 *alpha* sum(abs(w1))
  R2 = lambda0 *(1-alpha)*abs(t(w))%*%M%*%abs(w)/2 
  
  R1 + R2 - L
}

gelnet.logreg.obj2 = function(X,y,w,b, M,l1, l2){

  s <- X %*% w + b
  v <- y * s - log(1+exp(s))
  LL <- mean(v)

  R1 = l1*sum(abs(w))
  R2 = l2*t(w)%*%M%*%w/2
  R1 + R2 - LL
}
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# mian function
# Network-regularized Sparse Logistic Regression (NSLR) 
SGNLR = function(X, y, M, lambda0, alpha, niter=100, gate = 0){
  # -----------------
  # X   is a n x p matrix
  # y   is a n x 1 vector
  # M   is a (p+1) x (p+1) matrix
  # -----------------
  if(!is.matrix(y)){y = matrix(y,ncol = 1)}
  if(!is.matrix(X)){X = matrix(y,ncol = ncol(X))}
  
  n = dim(X)[1]; p = dim(X)[2]
  
  # Adjusting penalty parameters
  lambda = n*alpha*lambda0
  eta = n*(1-alpha)*lambda0
  
  X = cbind(X,rep(1,n))
  # X   is a n x (p+1) matrix
  
  if(dim(M)[1]!= p+1){M = cbind(M,rep(0,p)); M = rbind(M,rep(0,p+1))} 
  # M   is a (p+1) x (p+1) matrix
  # -----------------
  # Initialization
  w = matrix(0, nrow = p+1)
  err=0.0001
  
  obj=log.likelihhod = NULL
  for(i in 1:niter){
    w0 = w
    D = update.diagMatrix(X, w, gate)
    z = update.zVector(X, y, w, D)
    XDX = t(X)%*%D%*%X + eta*M
    t   = t(X)%*%D%*%z
    for(j in 1:p){
      w[j] = 0
      w[j] = (Soft.thresholding(t[j]-t(w)%*%XDX[j,], lambda))/XDX[j,j]
    }
    
    w[p+1] = 0
    w[p+1] = (t[p+1]-t(w)%*%XDX[p+1,])/XDX[p+1,p+1]
    
    # Calculate J (for testing convergence)
    log.likelihhod = c(log.likelihhod,objective(X,y,w))
    obj = c(obj,logreg.obj(X,y,w,M,lambda0, alpha))

    if(i>3&&abs(obj[i]-obj[i-1])/abs(obj[i-1])<0.00001){break}
  }
  names(w) = c(paste("w", 1:(length(w)-1), sep = ""), "Intercept")
  return(list(w=w, log.likelihhod = log.likelihhod,obj=obj))
}

# Absolute Network-regularized Logistic Regression (AbsNet.LR)
abs.SNLR = function(X, y, M, lambda0, alpha, niter=100, gate = 0){
  # -----------------
  # X   is a n x p matrix
  # y   is a n x 1 vector
  # M   is a (p+1) x (p+1) matrix
  # example:
  # -----------------
  if(!is.matrix(y)){y = matrix(y,ncol = 1)}
  if(!is.matrix(X)){X = matrix(y,ncol = ncol(X))}

  n = dim(X)[1]; p = dim(X)[2]

  # Adjusting penalty parameters
  lambda = n*alpha*lambda0
  eta = n*(1-alpha)*lambda0

  X = cbind(X,rep(1,n))
  # X   is a n x (p+1) matrix

  if(dim(M)[1]!= p+1){M = cbind(M,rep(0,p)); M = rbind(M,rep(0,p+1))}
  # M   is a (p+1) x (p+1) matrix
  # -----------------
  # Initialization
  w = matrix(0, nrow = p+1)
  err=0.0001

  obj = NULL
  for(i in 1:niter){
    w0 = w
    D = update.diagMatrix(X, w, gate)
    z = update.zVector(X, y, w, D)
    B = t(X)%*%D%*%X
    t   = t(X)%*%D%*%z
    for(j in 1:p){
      w[j] = 0
      soft.value = lambda + 2*eta*abs(t(w))%*%M[j,]
      w[j] = (Soft.thresholding(t[j]-t(w)%*%B[j,], soft.value))/(B[j,j]+2*eta*M[j,j])
    }
    XDX = B + eta*M
    w[p+1] = 0
    w[p+1] = (t[p+1]-t(w)%*%XDX[p+1,])/XDX[p+1,p+1]

    # Calculate J (for testing convergence)
    obj = c(obj,abs.logreg.obj(X,y,w,M,lambda0, alpha))

    if(i>3&&abs(obj[i]-obj[i-1])/abs(obj[i-1])<0.00001){break}
  }
  names(w) = c(paste("w", 1:(length(w)-1), sep = ""), "Intercept")
  return(list(w=w, obj = obj))
}
# ------------------------------------------------------------------
predict.SGNLR = function(Train.dat,True.label,Est.w){
  # -----------------
  # X   is a n x p matrix
  # w   is a (p+1) x 1 vector
  # w = (w1,w2,...,b)
  # -----------------
  library(pROC)
  X = Train.dat; y0 = True.label; w = Est.w
  n = dim(X)[1]; p = dim(X)[2]
  if(!is.matrix(w)){w = matrix(c(w),ncol = 1)}
  X = cbind(X,rep(1,n)) # X is a n x (p+1) matrix
  pp = 1/(1+exp(-X%*%w))
  y = rep(1,n)
  y[pp<0.5] = 0
  accuracy.rate = length(which(y==y0))/n 
  # AUC = accuracy.rate
  AUC = auc(y0, c(pp))[1]
  # AUC = roc(as.factor(y0), c(pp))$auc[1]
  
  return(list(pp=pp, y=y, acc = accuracy.rate, AUC = AUC))
}

