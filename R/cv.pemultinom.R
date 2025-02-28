#' Fit a multinomial regression model with Lasso penalty.
#'
#' Fit a multinomial regression model with Lasso penalty. This function implements the l1-penalized multinomial regression model (parameterized with a reference level). A cross-validation procedure is applied to choose the tuning parameter. See Tian et al. (2024) for details.
#' @export
#' @importFrom stats predict
#' @importFrom stats model.matrix
#' @importFrom stats relevel
#' @importFrom stats quantile
#' @importFrom stats pnorm
#' @importFrom stats qnorm
#' @importFrom stats sd
#' @importFrom stats cov
#' @importFrom utils tail
#' @importFrom utils globalVariables
#' @importFrom nnet multinom
#' @importFrom foreach foreach
#' @importFrom foreach %do%
#' @importFrom foreach %dopar%
#' @importFrom magrittr %>%
#' @importFrom doParallel registerDoParallel
#' @importFrom doParallel stopImplicitCluster
#' @importFrom lpSolve lp
#' @param x the design/predictor matrix, each row of which is an observation vector.
#' @param y the response variable. Can be of one type from factor/integer/character.
#' @param ref the reference level. Default = NULL, which sets the reference level to the last category (sorted by alphabetical order)
#' @param nfolds the number of cross-validation folds to use. Default = 5.
#' @param nlambda the number of penalty parameter candidates. Default = 100.
#' @param max_iter the maximum iteration rounds in each iteration of the coordinate descent algorithm. Default = 200.
#' @param tol convergence threshold (tolerance level) for coordinate descent. Default = 1e-3.
#' @param ncores the number of cores to use for parallel computing. Default = 1.
#' @param standardized logical flag for x variable standardization, prior to fitting the model sequence. Default = TRUE. Note that the fitting results will be translated to the original scale before output.
#' @param weights observation weights. Should be a vector of non-negative numbers of length n (the number of observations). Default = NULL, which sets equal weights for all observations.
#' @param info whether to print the information or not. Default = TRUE.
#' @param lambda_min_ratio the ratio between lambda.min and lambda.max, where lambda.max is automatically determined by the code and the lambda sequence will be determined by `exp(seq(log(lambda.max), log(lambda.min), len = nlambda))`.
#' @return A list with the following components.
#' \item{beta.list}{the estimates of coefficients. It is a list of which the k-th component is the contrast coefficient between class k and the reference class corresponding to different lambda values. The j-th column of each list component corresponds to the j-th lambda value.}
#' \item{beta.1se}{the coefficient estimate corresponding to lambda.1se. It is a matrix, and the k-th column is the contrast coefficient between class k and the reference class.}
#' \item{beta.min}{the coefficient estimate corresponding to lambda.min. It is a matrix, and the k-th column is the contrast coefficient between class k and the reference class.}
#' \item{lambda.1se}{the largest value of lambda such that error is within 1 standard error of the minimum. See Chapter 2.3 of Hastie et al. (2015) for more details.}
#' \item{lambda.min}{the value of lambda that gives minimum cvm.}
#' \item{cvm}{the weights in objective function.}
#' \item{cvsd}{the estimated marginal probability for each class.}
#' \item{lambda}{lambda values considered in the cross-validation process.}
#' @seealso \code{\link{debiased_lasso}}, \code{\link{predict_pemultinom}}.
#' @references
#' Hastie, T., Tibshirani, R., & Wainwright, M. (2015). Statistical learning with sparsity. Monographs on statistics and applied probability, 143.
#'
#' Tian, Y., Rusinek, H., Masurkar, A. V., & Feng, Y. (2024). L1‐Penalized Multinomial Regression: Estimation, Inference, and Prediction, With an Application to Risk Factor Identification for Different Dementia Subtypes. Statistics in Medicine, 43(30), 5711-5747.
#'
#' @useDynLib pemultinom
#' @import Rcpp
#'
#' @examples
#' # generate data from a logistic regression model with n = 100, p = 50, and K = 3
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' n <- 100
#' p <- 50
#' K <- 3
#'
#' Sigma <- outer(1:p, 1:p, function(x,y) {
#'   0.9^(abs(x-y))
#' })
#' R <- chol(Sigma)
#' s <- 3
#' beta_coef <- matrix(0, nrow = p+1, ncol = K-1)
#' beta_coef[1+1:s, 1] <- c(3, 3, 3)
#' beta_coef[1+1:s+s, 2] <- c(3, 3, 3)
#'
#' x <- matrix(rnorm(n*p), ncol = p) %*% R
#' y <- sapply(1:n, function(j){
#'   prob_i <- c(sapply(1:(K-1), function(k){
#'     exp(sum(x[j, ]*beta_coef[-1, k]))
#'   }), 1)
#'   prob_i <- prob_i/sum(prob_i)
#'   sample(1:K, size = 1, replace = TRUE, prob = prob_i)
#' })
#'
#' # fit the l1-penalized multinomial regression model
#' fit <- cv.pemultinom(x, y, ncores = 2)
#' beta <- fit$beta.min
#'
#' # generate test data from the same model
#' x.test <- matrix(rnorm(n*p), ncol = p) %*% R
#' y.test <- sapply(1:n, function(j){
#'   prob_i <- c(sapply(1:(K-1), function(k){
#'     exp(sum(x.test[j, ]*beta_coef[-1, k]))
#'   }), 1)
#'   prob_i <- prob_i/sum(prob_i)
#'   sample(1:K, size = 1, replace = TRUE, prob = prob_i)
#' })
#'
#' # predict labels of test data and calculate the misclassification error rate (using beta.min)
#' ypred.min <- predict_pemultinom(fit$beta.min, ref = 3, xnew = x.test, type = "class")
#' mean(ypred.min != y.test)
#'
#' # predict labels of test data and calculate the misclassification error rate (using beta.1se)
#' ypred.1se <- predict_pemultinom(fit$beta.1se, ref = 3, xnew = x.test, type = "class")
#' mean(ypred.1se != y.test)
#'
#' # predict posterior probabilities of test data
#' ypred.prob <- predict_pemultinom(fit$beta.min, ref = 3, xnew = x.test, type = "prob")

cv.pemultinom <- function(x, y, ref = NULL, nfolds = 5, nlambda = 100, max_iter = 200, tol = 1e-3, ncores = 1,
                          standardized = TRUE, weights = NULL, info = TRUE, lambda_min_ratio = 0.01) {
  p <- ncol(x)
  n <- nrow(x)
  K <- length(unique(y))

  y.ori <- y
  y <- numeric(length(y.ori))
  if (is.null(ref)) {
    ref <- tail(sort(unique(y.ori)), 1)
    if (info) {
      message(paste("No reference level is specified! Class '", ref, "' is set as the reference class.", sep = ""))
    }
  }

  y.value <- sort(unique(y.ori[y.ori != ref]))
  for (k in 1:(K-1)) {
    y[y.ori == y.value[k]] <- k
  }
  y[y.ori == ref] <- K

  # just added----
  if (is.null(weights)) {
    weights <- rep(1, n)
  }
  # just added----

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


  # fit the model for all lambdas by using all the data
  # standardization
  if (standardized) {
    x.scaled <- scale(x)
    x.backup <- x
    x.center <- attr(x.scaled,"scaled:center")
    x.std <- attr(x.scaled,"scaled:scale")
    zero_ind <- (abs(x.std) <= 1e-16)
    x.backup[, !zero_ind] <- x.scaled[, !zero_ind]
    x.scaled <- x.backup
    x.std[zero_ind] <- 1
  } else {
    x.scaled <- x
    x.center <- rep(0, p)
    x.std <- rep(1, p)
    zero_ind <- rep(FALSE, p)
  }

  y.dummy <- as.matrix(model.matrix(~ factor(y)-1))
  lambda_list <- set_lambda(x = x.scaled, y = y, nlambda = nlambda, weights = weights, lambda_min_ratio = lambda_min_ratio)
  L <- pemultinom_c_reverse(x = x.scaled, y = y.dummy, lambda_list = lambda_list, max_iter = max_iter, tol = tol,
                            zero_ind = zero_ind, weights = weights, intercept = TRUE)

  beta <- sapply(1:(K-1), function(l){
    beta_cur <- matrix(L[[1]][, l, drop = FALSE]/x.std, nrow = p)
    beta_combined <- rbind(L[[2]][, l] - x.center %*% beta_cur, beta_cur)
    if (all(!is.null(colnames(x)))) {
      rownames(beta_combined) <- c("Intercept", colnames(x))
    } else {
      rownames(beta_combined) <- c("Intercept", paste0("X", 1:p))
    }
    beta_combined
  }, simplify = FALSE)

  names(beta) <- y.value

  # cross-validation
  if (ncores > 1) {
    registerDoParallel(ncores)
    fit_cv <- foreach(k = 1:nfolds) %dopar% {
      x_train <- x[fold != k, ]
      y_train <- y[fold != k]
      x_valid <- x[fold == k, ]
      y_valid <- y[fold == k]

      if (standardized) {
        x.scaled <- scale(x_train)
        x.backup <- x_train
        x.center <- attr(x.scaled,"scaled:center")
        x.std <- attr(x.scaled,"scaled:scale")
        zero_ind <- (abs(x.std) <= 1e-16)
        x.backup[, !zero_ind] <- x.scaled[, !zero_ind]
        x.scaled <- x.backup
        x.std[zero_ind] <- 1
      } else {
        x.scaled <- x_train
        x.center <- rep(0, p)
        x.std <- rep(1, p)
      }

      y.dummy <- as.matrix(model.matrix(~ factor(y_train)-1))

      L <- pemultinom_c_reverse(x = x.scaled, y = y.dummy, lambda_list = lambda_list, max_iter = max_iter, tol = tol,
                                zero_ind = zero_ind, weights = weights[fold != k], intercept = TRUE)

      beta_cv <- sapply(1:(K-1), function(l){
        beta_cur <- matrix(L[[1]][, l, drop = FALSE]/x.std, nrow = p)
        beta_combined <- rbind(L[[2]][, l] - x.center %*% beta_cur, beta_cur)
        if (all(!is.null(colnames(x)))) {
          rownames(beta_combined) <- c("Intercept", colnames(x))
        } else {
          rownames(beta_combined) <- c("Intercept", paste0("X", 1:p))
        }
        beta_combined
      }, simplify = FALSE)
      beta_cv
    }

    loss_lambda <- sapply(1:nlambda, function(r){
      sapply(1:nfolds, function(k){
        beta <- sapply(1:(K-1), function(l){
          fit_cv[[k]][[l]][, r]
        })
        prob_pred <- predict_pemultinom(beta = beta, xnew = x[fold == k, ], type = "prob", ref = K)
        prob_pred[prob_pred <= 1e-10] <- 1e-10
        loss(prob_pred, y[fold == k], weights = weights[fold == k])
      })
    })
  } else {
    fit_cv <- foreach(k = 1:nfolds) %do% {
      x_train <- x[fold != k, ]
      y_train <- y[fold != k]
      x_valid <- x[fold == k, ]
      y_valid <- y[fold == k]

      if (standardized) {
        x.scaled <- scale(x_train)
        x.backup <- x_train
        x.center <- attr(x.scaled,"scaled:center")
        x.std <- attr(x.scaled,"scaled:scale")
        zero_ind <- (abs(x.std) <= 1e-16)
        x.backup[, !zero_ind] <- x.scaled[, !zero_ind]
        x.scaled <- x.backup
        x.std[zero_ind] <- 1
      } else {
        x.scaled <- x_train
        x.center <- rep(0, p)
        x.std <- rep(1, p)
      }
      y.dummy <- as.matrix(model.matrix(~ factor(y_train)-1))

      L <- pemultinom_c_reverse(x = x.scaled, y = y.dummy, lambda_list = lambda_list, max_iter = max_iter, tol = tol,
                                zero_ind = zero_ind, weights = weights[fold != k], intercept = TRUE)

      beta_cv <- sapply(1:(K-1), function(l){
        beta_cur <- matrix(L[[1]][, l, drop = FALSE]/x.std, nrow = p)
        beta_combined <- rbind(L[[2]][, l] - x.center %*% beta_cur, beta_cur)
        if (all(!is.null(colnames(x)))) {
          rownames(beta_combined) <- c("Intercept", colnames(x))
        } else {
          rownames(beta_combined) <- c("Intercept", paste0("X", 1:p))
        }
        beta_combined
      }, simplify = FALSE)
      beta_cv
    }

    loss_lambda <- sapply(1:nlambda, function(r){
      sapply(1:nfolds, function(k){
        beta <- sapply(1:(K-1), function(l){
          fit_cv[[k]][[l]][, r]
        })
        prob_pred <- predict_pemultinom(beta = beta, xnew = x[fold == k, ], type = "prob", ref = K)
        prob_pred[prob_pred <= 1e-10] <- 1e-10
        loss(prob_pred, y[fold == k], weights = weights[fold == k])
      })
    })
  }

  cvm <- colMeans(loss_lambda, na.rm = TRUE)
  cvsd <- apply(loss_lambda, 2, function(x){sd(x, na.rm = TRUE)})
  ind.min <- which.min(cvm)
  lambda.min <- lambda_list[ind.min]
  cvsd.min <- cvsd[ind.min]
  cvm.min <- cvm[ind.min]
  ind.1se <- min(which(cvm <= cvm.min + cvsd.min))
  lambda.1se <- lambda_list[ind.1se]
  ind.3se <- max(which(cvm <= cvm.min + cvsd.min))
  lambda.3se <- lambda_list[ind.3se]

  beta.1se <- sapply(1:(K-1), function(k){
    beta[[k]][, ind.1se]
  })

  colnames(beta.1se) <- y.value

  beta.min <- sapply(1:(K-1), function(k){
    beta[[k]][, ind.min]
  })

  colnames(beta.min) <- y.value

  beta.3se <- sapply(1:(K-1), function(k){
    beta[[k]][, ind.3se]
  })

  colnames(beta.3se) <- y.value


  if (standardized) {
    beta.1se[-1, ] <- beta.1se[-1, ]/x.std
    beta.min[-1, ] <- beta.min[-1, ]/x.std
    beta.3se[-1, ] <- beta.3se[-1, ]/x.std

    beta.1se[1, ] <- beta.1se[1, ] - t(beta.1se[-1, ]) %*% x.center
    beta.min[1, ] <- beta.min[1, ] - t(beta.min[-1, ]) %*% x.center
    beta.3se[1, ] <- beta.3se[1, ] - t(beta.3se[-1, ]) %*% x.center

    lambda_list <- lambda_list*x.std
    lambda.1se <- lambda.1se*x.std
    lambda.min <- lambda.min*x.std
  }


  stopImplicitCluster()

  # return(list(beta.list = beta, beta.1se = beta.1se, beta.min = beta.min, lambda.1se = lambda.1se, lambda.min = lambda.min,
  #             cvm = cvm, cvsd = cvsd, lambda = lambda_list, weights = weights, x.scaled = x.scaled, x.center = x.center,
  #             x.std = x.std, beta.scaled.min = beta.scaled.min))
  return(list(beta.list = beta, beta.1se = beta.1se, beta.min = beta.min, beta.3se = beta.3se, lambda.1se = lambda.1se,
              lambda.min = lambda.min, lambda.3se = lambda.3se, cvm = cvm, cvsd = cvsd, lambda = lambda_list))
}





