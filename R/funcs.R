

loss <- function(prob, y_valid, weights) {
  -sum(weights*log(prob[cbind(1:length(y_valid), y_valid)]))
}


set_lambda <- function(x, y, ref = NULL, nlambda = 100, weights) {
  K <- length(unique(y))
  y.leveled <- relevel(as.factor(y), ref = as.character(K))
  fit_nnet <- multinom(y.leveled~1, trace = FALSE, weights = weights)
  y.dummy <- as.matrix(model.matrix(~ y.leveled-1))
  if (K == 2) {
    res <- y.dummy[, 2]-fit_nnet$fitted.values
    zmax <- max(t(x) %*% res)/nrow(x)

  } else {
    res <- y.dummy-fit_nnet$fitted.values
    zmax <- max(sapply(2:K, function(j){
      max(t(x) %*% res[, j])/nrow(x)
    }))
  }


  lambda.max <- zmax
  lambda.min <- 0.01*lambda.max
  lambda <- exp(seq(log(lambda.max), log(lambda.min), len = nlambda))
  return(lambda)
}


inv_cov_calc <- function(x, y, beta, nfolds = 5, nlambda = 100, max_iter = 200, tol = 1e-3, ncores = 1,
                         lambda.choice = c("lambda.1se", "lambda.min")) {
  registerDoParallel(ncores)

  pb_calc <- function(X, beta) {
    expxb <- cbind(exp(cbind(1, X) %*% beta), 1)
    pb <- expxb/rowSums(expxb)
    pb
  }

  p <- ncol(x)
  n <- nrow(x)
  K <- length(unique(y))
  p.big <- (K-1)*p

  n_list <- as.numeric(table(y))
  fold_by_class <- sapply(1:K, function(k){
    lb <- sample(1:n_list[k] %% nfolds)
    lb[lb == 0] <- nfolds
    lb
  }, simplify = F)
  fold <- numeric(n)
  for (k in 1:K) {
    fold[y == k] <- fold_by_class[[k]]
  }

  pb <- pb_calc(X=x, beta = beta)
  Sigma.full <- matrix(nrow = (K-1)*p, ncol=(K-1)*p)
  for (k1 in 1:(K-1)) {
    for (k2 in 1:(K-1)) {
      if (k1 == k2) {
        Sigma.full[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x) %*% diag(pb[,k1]*(1-pb[,k1])) %*% x)/n
      } else {
        Sigma.full[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x) %*% diag(-pb[,k1]*pb[,k2]) %*% x)/n
      }
    }
  }
  lambda.max <- sapply(1:p.big, function(j){
    max(Sigma.full[j, -j])
  })


  loss <- sapply(1:nfolds, function(r){
    message(paste("Fold ", r, " of 5......", sep = ""))

    pb <- pb_calc(X=x[fold != r, ], beta = beta)
    Sigma <- matrix(nrow = (K-1)*p, ncol=(K-1)*p)
    for (k1 in 1:(K-1)) {
      for (k2 in 1:(K-1)) {
        if (k1 == k2) {
          Sigma[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x[fold != r, ]) %*% diag(pb[,k1]*(1-pb[,k1])) %*% x[fold != r, ])/sum(fold != r)
        } else {
          Sigma[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x[fold != r, ]) %*% diag(-pb[,k1]*pb[,k2]) %*% x[fold != r, ])/sum(fold != r)
        }
      }
    }
    Sigma.valid <- matrix(nrow = (K-1)*p, ncol=(K-1)*p)
    pb <- pb_calc(X=x[fold == r, ], beta = beta)
    for (k1 in 1:(K-1)) {
      for (k2 in 1:(K-1)) {
        if (k1 == k2) {
          Sigma.valid[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x[fold == r, ]) %*% diag(pb[,k1]*(1-pb[,k1])) %*% x[fold == r, ])/sum(fold == r)
        } else {
          Sigma.valid[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x[fold == r, ]) %*% diag(-pb[,k1]*pb[,k2]) %*% x[fold == r, ])/sum(fold == r)
        }
      }
    }


    loss <- foreach(j = 1:p.big, .combine = "rbind") %dopar% {
      if (lambda.max[j] == 0) {
        rep(NA, nlambda)
      } else {
        lambda.min <- 0.01*lambda.max[j]
        lambda_list <- exp(seq(log(lambda.max[j]), log(lambda.min), len = nlambda))
        gamma_j_matrix <- penalized_quad_r(A = Sigma[-j, -j], b = Sigma[j, -j], lambda_list = lambda_list, max_iter = max_iter, tol = tol)
        sapply(1:nlambda, function(l){
          Sigma.valid[j,j] - Sigma.valid[j, -j] %*% gamma_j_matrix[l, ] + gamma_j_matrix[l, ] %*% Sigma.valid[-j, -j] %*% gamma_j_matrix[l, ]/2
        })
      }
    }
    loss
  }, simplify = FALSE)
  message("Determining the optimal lambda and finalizing the inference results......")
  # calculate gamma for each j with the full data and best lambda
  gamma_matrix <- foreach(j = 1:p.big, .combine = "rbind") %dopar% {
    if (lambda.max[j] == 0) {
      rep(0, p.big - 1)
    } else {
      lambda.min <- 0.01*lambda.max[j]
      lambda_list <- exp(seq(log(lambda.max[j]), log(lambda.min), len = nlambda))
      gamma_j_matrix <- penalized_quad_r(A = Sigma.full[-j, -j], b = Sigma.full[j, -j], lambda_list = lambda_list, max_iter = max_iter, tol = tol)
      loss_j <- sapply(1:nfolds, function(r){
        loss[[r]][j,]
      })
      cvm <- rowMeans(loss_j, na.rm = TRUE)
      cvsd <- apply(loss_j, 1, function(x){sd(x, na.rm = TRUE)})
      ind.min <- which.min(cvm)
      lambda.min <- lambda_list[ind.min]
      cvsd.min <- cvsd[ind.min]
      cvm.min <- cvm[ind.min]
      ind.1se <- min(which(cvm <= cvm.min + cvsd.min))
      lambda.1se <- lambda_list[ind.1se]
      if(lambda.choice == "lambda.1se") {
        gamma_j <- gamma_j_matrix[ind.1se, ]
      } else if(lambda.choice == "lambda.min") {
        gamma_j <- gamma_j_matrix[ind.min, ]
      }
      gamma_j
    }
  }

  tau2 <- sapply(1:p.big, function(j){
    Sigma.full[j, j] - Sigma.full[j,-j] %*% gamma_matrix[j, ]
  })

  Theta <- matrix(nrow = p.big, ncol = p.big)
  for (j in 1:p.big) {
    if (tau2[j] == 0){
      Theta[j, ] <- 0
    } else {
      Theta[j, j] <- 1/tau2[j]
      Theta[j, -j] <- -gamma_matrix[j, ]/tau2[j]
    }

  }

  stopImplicitCluster()

  return(list(Theta = Theta, const.index = which(lambda.max == 0)))
}


penalized_quad_r <- function(A, b, lambda_list, max_iter=200, tol=1e-3) {
  penalized_quad(A, b, lambda_list, max_iter, tol)
}



multisplits <- function(x, y, lambda = c("lambda.1se", "lambda.min"), ncores = 1, B = 50) {
  lambda <- match.arg(lambda)
  K <- length(unique(y))
  n <- length(y)
  p <- ncol(x)

  p_value_list <- rep(list(matrix(1,nrow = p+1, ncol = B)), K)
  y.fac <- relevel(as.factor(y), ref = K)
  for (b in 1:B) {
    # print(b)
    # set.seed(b+10, kind = "L'Ecuyer-CMRG")
    while (1) {
      ind <- sample(n, size = floor(n/2))

      fit_lasso <- try(cv.pemultinom(x = x[ind, ], y = y[ind], ncores = ncores))
      if (!inherits(fit_lasso, "try-error")) {
        if (lambda == "lambda.1se") {
          S_active <- which(rowSums(fit_lasso$beta.1se[-1, ] != 0) > 0)
        } else if (lambda == "lambda.min") {
          S_active <- which(rowSums(fit_lasso$beta.min[-1, ] != 0) > 0)
        }

        if (length(S_active) > 0) {
          fit_ols <- multinom(y.fac~x[, S_active, drop = FALSE], trace = FALSE)
          z <- summary(fit_ols)$coefficients/summary(fit_ols)$standard.errors
          pvalue <- (1 - pnorm(abs(z))) * 2
        } else {
          fit_ols <- multinom(y.fac~1, trace = FALSE)
          z <- summary(fit_ols)$coefficients/summary(fit_ols)$standard.errors
          pvalue <- (1 - pnorm(abs(z))) * 2
        }

        if (all(is.finite(summary(fit_ols)$standard.errors))) {
          break
        }
      }

    }

    for (k in 1:(K-1)) {
      p_value_list[[k]][c(1, 1+S_active), b] <- sapply(1:NCOL(pvalue), function(j){
        min(pvalue[k,j]*length(S_active), 1)
      })
    }
  }


  p_value <- sapply(1:(K-1), function(k){
    sapply(1:(p+1), function(j){
      Qj <- sapply(seq(0.05, 1, 0.01), function(gamma){
        min(1, quantile(p_value_list[[k]][j, ]/gamma, gamma, na.rm = TRUE))
      })
      min(1, (1-log(0.05))*min(Qj))
    })
  })

  return(p_value)
}


bootstrap_multinom <- function(x, y, lambda = c("lambda.1se", "lambda.min"), ncores = 1, B = 50, alpha = 0.05,
                               type = c("vector", "residual")) {
  lambda <- match.arg(lambda)
  type <- match.arg(type)
  K <- length(unique(y))
  n <- length(y)
  p <- ncol(x)

  L <- rep(list(matrix(1,nrow = p+1, ncol = B)), K)

  if (type == "vector") {
    for (b in 1:B) {
      while (1) {
        y_b <- 1
        while (length(unique(y_b)) != K) {
          ind_b <- sample(n, size = n, replace = TRUE)
          y_b <- y[ind_b]
          x_b <- x[ind_b, ]
        }
        fit_b <- try(cv.pemultinom(x = x_b, y = y_b, ncores = ncores))
        if (!inherits(fit_b, "try-error")) {
          if (lambda == "lambda.1se") {
            for (k in 1:(K-1)) {
              L[[k]][, b] <- fit_b$beta.1se[, k]
            }
          } else if (lambda == "lambda.min") {
            for (k in 1:(K-1)) {
              L[[k]][, b] <- fit_b$beta.min[, k]
            }
          }
          break
        }
      }
    }
  } else if (type == "residual") {
    fit_initial <- cv.pemultinom(x = x, y = y, ncores = ncores)
    if (lambda == "lambda.1se") {
      beta <- fit_initial$beta.1se
    } else if (lambda == "lambda.min") {
      beta <- fit_initial$beta.min
    }

    prob <- predict_pemultinom(beta = beta, ref = K, xnew = x, type = "prob")

    for (b in 1:B) {
      while (1) {
        y_b <- 1
        while (length(unique(y_b)) != K) {
          y_b <- sapply(1:n, function(i){
            sample(1:K, size = 1, prob = prob[i, ])
          })
        }
        fit_b <- try(cv.pemultinom(x = x, y = y_b, ncores = ncores))
        if (!inherits(fit_b, "try-error")) {
          if (lambda == "lambda.1se") {
            for (k in 1:(K-1)) {
              L[[k]][, b] <- fit_b$beta.1se[, k]
            }
          } else if (lambda == "lambda.min") {
            for (k in 1:(K-1)) {
              L[[k]][, b] <- fit_b$beta.min[, k]
            }
          }
          break
        }
      }

    }
  }

  CI <- sapply(1:(K-1), function(k){
    t(sapply(1:(p+1), function(j){
      c(quantile(L[[k]][j, ], alpha/2, na.rm = TRUE), quantile(L[[k]][j, ], 1-alpha/2, na.rm = TRUE))
    }))
  }, simplify = FALSE)


  return(CI)
}
