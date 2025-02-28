#' Doing statistical inference on l1-penalized multinomial regression via debiased Lasso (or desparisified Lasso).
#'
#' Doing statistical inference on l1-penalized multinomial regression via debiased Lasso (or desparisified Lasso). This function implements the algorithm described in Tian et al. (2024), which is an extension of the original debiased Lasso (Van de Geer et al. 2014; Zhang and Zhang. 2014; Javanmard, A. and Montanari, A. (2014)) to the multinomial case.
#' @export
#' @param x the design/predictor matrix, each row of which is an observation vector.
#' @param y the response variable. Can be of one type from factor/integer/character.
#' @param ref the reference level. Default = NULL, which sets the same reference level as used in obtaining \code{beta}. Even when the user inputs \code{ref} manually, it should be always the same reference level as used in obtaining \code{beta}.
#' @param beta the beta estimate from l1-penalized multinomial regression. Should be in the same format as \code{beta.min} or \code{beta.1se} in output of function \code{\link{cv.pemultinom}}. The user is recommended to directly pass \code{beta.min} or \code{beta.1se} from the output of function \code{\link{cv.pemultinom}} to parameter \code{beta}.
#' @param nfolds the number of cross-validation folds to use. Cross-validation is used to determine the best tuning parameter lambda in the nodewise regression, i.e., Algorithm 2 in Tian et al. (2024). Default = 5.
#' @param ncores the number of cores to use for parallel computing. Default = 1.
#' @param nlambda the number of penalty parameter candidates in the cross-validation procedure. Cross-validation is used to determine the best tuning parameter lambda in the nodewise regression, i.e., Algorithm 2 in Tian et al. (2024). Default = 100.
#' @param max_iter the maximum iteration rounds in each iteration of the coordinate descent algorithm. Default = 200.
#' @param tol convergence threshold (tolerance level) for coordinate descent. Default = 1e-3.
#' @param lambda.choice the choice of the tuning parameter lambda used in the nodewise regression. It can only be either "lambda.min" or "lambda.1se". The interpretation is the same as in the `cv.pemultinom` function. Default = "lambda.min".
#' @param alpha significance level used in the output confidence interval. Has to be a number between 0 and 1. Default = 0.05.
#' @param method which method to estimate the Hessian inverse matrix. Can be either "nodewise_reg" or "LP".
#' \itemize{
#' \item nodewise_reg: the method presented in the main text of Tian et al. (2024).
#' \item LP: the method presented in Section A of the supplements of Tian et al. (2024). Warning: This method is not well implemented as `eta` parameter has to be set as fixed and currently cannot be automatically tuned.
#' }
#' @param info whether to print the information or not. Default = TRUE.
#' @param LP.parameter If \code{method} = `LP`, then this parameter will be used. It has to be a list of two components like `list(eta = NULL, split.ratio = 0.5)`. Here is the interpretation:
#' \itemize{
#' \item eta: This is the parameter used in the LP method which is the righthand side of the contraints. Default `eta` = `NULL`, which will be set as \eqn{2\sqrt{\log(p)/n}} where p is the number of features and n is the total sample size.
#' \item split.ratio: The split ratio used in the LP method. The data will be splitted into two parts by class based on `split.ratio`. The first part will be used to fit the Lasso estimator, and the second part will be used to fit the Hessian inverse through the LP method. Default `split.ratio` = 0.5.
#' }
#' @param lambda_min_ratio the ratio between lambda.min and lambda.max used for the Lasso problem in the nodewise regression (for estimating the Hessian inverse), where lambda.max is automatically determined by the code and the lambda sequence will be determined by `exp(seq(log(lambda.max), log(lambda.min), len = nlambda))`.
#' @return A list of data frames, each of which contains the inference results for each class (v.s. the reference class). In the data frame, each row represents a variable. The columns include:
#' \item{beta}{the debiased point estimate of the coefficient}
#' \item{p_value}{p value of each variable}
#' \item{CI_lower}{lower endpoint of the confidence interval for each coefficient}
#' \item{CI_upper}{upper endpoint of the confidence interval for each coefficient}
#' \item{std_dev}{the estimated standard deviation of each component of beta estimate}
#' @seealso \code{\link{cv.pemultinom}}, \code{\link{predict_pemultinom}}.
#' @references
#' Tian, Y., Rusinek, H., Masurkar, A. V., & Feng, Y. (2024). L1‐Penalized Multinomial Regression: Estimation, Inference, and Prediction, With an Application to Risk Factor Identification for Different Dementia Subtypes. Statistics in Medicine, 43(30), 5711-5747.
#'
#' Van de Geer, S., Bühlmann, P., Ritov, Y. A., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. The Annals of Statistics, 42(3), 1166-1202.
#'
#' Zhang, C. H., & Zhang, S. S. (2014). Confidence intervals for low dimensional parameters in high dimensional linear models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(1), 217-242.
#'
#' Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. The Journal of Machine Learning Research, 15(1), 2869-2909.
#' @examples
#' \donttest{
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
#' # run the debiasing approach
#' fit_debiased <- debiased_lasso(x, y, beta = beta, ncores = 2)
#' }


debiased_lasso <- function(x, y, ref = NULL, beta = NULL, nfolds = 5, ncores = 1, nlambda = 50, max_iter = 200, tol = 1e-3,
                           lambda.choice = "lambda.min", alpha = 0.05, method = c("nodewise_reg", "LP"), info = TRUE,
                           LP.parameter = list(eta = NULL, split.ratio = 0.5), lambda_min_ratio = 0.01) {
  K <- length(unique(y))
  method <- match.arg(method)
  p <- ncol(x)
  n <- nrow(x)
  K <- length(unique(y))
  p.big <- (K-1)*p

  y.ori <- y
  y <- numeric(length(y.ori))
  if (is.null(ref)) {
    ref <- setdiff(unique(y.ori), colnames(beta))
    if (info) {
      message(paste("No reference level is specified! Class '", ref, "' is automatically detected from beta input as the reference class.", sep = ""))
    }
  }

  y.value <- sort(unique(y.ori[y.ori != ref]))
  for (k in 1:(K-1)) {
    y[y.ori == y.value[k]] <- k
  }
  y[y.ori == ref] <- K
  if (method == "nodewise_reg") {
    Theta_list <- inv_cov_calc(x = x, y = y, beta = beta, nfolds = nfolds, nlambda = nlambda, max_iter = max_iter,
                               tol = tol, ncores = ncores, lambda.choice = lambda.choice, lambda_min_ratio = lambda_min_ratio,
                               r = 0.1)
    Theta <- Theta_list$Theta
  } else if (method == "LP") {
    if (is.null(LP.parameter$eta)) {
      LP.parameter$eta <- 2*sqrt(log(p)/n)
    }

    # split data
    n_list <- as.numeric(table(y))

    ind <- Reduce("c", sapply(1:K, function(k){
      sample(n_list[k], floor(n_list[k]*LP.parameter$split.ratio))
    }, simplify = F))
    x1 <- x[ind, ]
    y1 <- y[ind]
    x2 <- x[-ind, ]
    y2 <- y[-ind]

    # fit beta using Lasso with (x1, y1)
    fit <- cv.pemultinom(x1, y1, nfolds = 5, nlambda = 100, ncores = ncores, max_iter = 200)
    if (lambda.choice == "lambda.min") {
      beta <- fit$beta.min
    }
    Theta <- M_LP(x2, y2, beta, LP.parameter$eta, ncores)

    # replace (x, y) with (x1, y1) as we will only use (x1, y1) below for the LP method
    x <- x1
    y <- y1
    n <- NROW(x)
  }

  y.dummy <- sapply(1:(K-1), function(k){
    1*I(y == k)
  })


  pb_calc <- function(X, beta) {
    expxb <- cbind(exp(cbind(1, X) %*% beta), 1)
    pb <- expxb/rowSums(expxb)
    pb
  }
  p <- ncol(x)
  p.big <- (K-1)*p

  pb <- pb_calc(X = x, beta = beta)
  epsilon <- sapply(1:(K-1), function(k){
    y.dummy[, k] - pb[, k]
  })
  Xtepsilon <- as.numeric(sapply(1:(K-1), function(k){
    t(x) %*% epsilon[, k]
  }))
  b <- as.numeric(beta[-1, ]) + Theta %*% Xtepsilon/n

  if (method == "nodewise_reg") {
    b[Theta_list$const.index] <- 0
  }

  cov.hat <- matrix(nrow = (K-1)*p, ncol=(K-1)*p)
  cov.choice <- 1
  if (cov.choice == 1) {
    for (k1 in 1:(K-1)) {
      for (k2 in 1:(K-1)) {
        if (k1 == k2) {
          cov.hat[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x) %*% diag(pb[,k1]*(1-pb[,k1])) %*% x)/n
        } else {
          cov.hat[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x) %*% diag(-pb[,k1]*pb[,k2]) %*% x)/n
        }
      }
    }
  } else {
    cov.hat <- cov(Reduce(cbind, sapply(1:(K-1), function(k){
      x*epsilon[, k]
    }, simplify = F)))
  }


  sigma2 <- sapply(1:p.big, function(j){
    t(Theta[j, ]) %*% cov.hat %*% Theta[j, ]
  })

  if (method == "nodewise_reg") {
    sigma2[Theta_list$const.index] <- Inf
  }

  test.stats.value <- sqrt(n)*b/sqrt(sigma2)
  CI <- data.frame(beta = b, p_value = 2*pnorm(-abs(test.stats.value)), CI_lower = b-qnorm(1-alpha/2)*sqrt(sigma2)/sqrt(n), CI_upper = b+qnorm(1-alpha/2)*sqrt(sigma2)/sqrt(n), z = sqrt(n)*b/sqrt(sigma2), std_dev = sqrt(sigma2)/sqrt(n))

  if (method == "nodewise_reg") {
    CI$std_dev[Theta_list$const.index] <- NA
  }

  CI <- sapply(1:(K-1), function(k){
    df <- CI[(1+(k-1)*p):(k*p), ]
    if (all(!is.null(colnames(x)))) {
      rownames(df) <- colnames(x)
    } else {
      rownames(df) <- paste0("X", 1:p)
    }
    df
  }, simplify = FALSE)

  names(CI) <- y.value

  return(CI)
}
