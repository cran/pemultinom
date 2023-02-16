#' Doing statistical inference on l1-penalized multinomial regression via debiased Lasso (or desparisified Lasso).
#'
#' Doing statistical inference on l1-penalized multinomial regression via debiased Lasso (or desparisified Lasso). This function implements the algorithm described in Tian et al. (2023), which is an extension of the original debiased Lasso (Van de Geer et al. (2014); Zhang and Zhang (2014)) to the multinomial case.
#' @export
#' @param x the design/predictor matrix, each row of which is an observation vector.
#' @param y the response variable. Can be of one type from factor/integer/character.
#' @param ref the reference level. Default = NULL, which sets the same reference level as used in obtaining \code{beta}. Even when the user inputs \code{ref} manually, it should be always the same reference level as used in obtaining \code{beta}.
#' @param beta the beta estimate from l1-penalized multinomial regression. Should be in the same format as \code{beta.min} or \code{beta.1se} in output of function \code{\link{cv.pemultinom}}. The user is recommended to directly pass \code{beta.min} or \code{beta.1se} from the output of function \code{\link{cv.pemultinom}} to parameter \code{beta}.
#' @param nfolds the number of cross-validation folds to use. Cross-validation is used to determine the best tuning parameter lambda in the nodewise regression, i.e., Algorithm 2 in Tian et al. (2023). Default = 5.
#' @param ncores the number of cores to use for parallel computing. Default = 1.
#' @param nlambda the number of penalty parameter candidates in the cross-validation procedure. Cross-validation is used to determine the best tuning parameter lambda in the nodewise regression, i.e., Algorithm 2 in Tian et al. (2023). Default = 100.
#' @param max_iter the maximum iteration rounds in each iteration of the coordinate descent algorithm. Default = 200.
#' @param tol convergence threshold (tolerance level) for coordinate descent. Default = 1e-3.
#' @param lambda.choice observation weights. Should be a vector of non-negative numbers of length n (the number of observations). Default = NULL, which sets equal weights for all observations.
#' @param alpha significance level used in the output confidence interval. Has to be a number between 0 and 1. Default = 0.05.
#' @return A list of data frames, each of which contains the inference results for each class (v.s. the reference class). In the data frame, each row represents a variable. The columns include:
#' \item{beta}{the debiased point estimate of the coefficient}
#' \item{p_value}{p value of each variable}
#' \item{CI_lower}{lower endpoint of the confidence interval for each coefficient}
#' \item{CI_upper}{upper endpoint of the confidence interval for each coefficient}
#' \item{std_dev}{the estimated standard deviation of each component of beta estimate}
#' @seealso \code{\link{cv.pemultinom}}, \code{\link{predict_pemultinom}}.
#' @references
#' Tian, Y., Rusinek, H., Masurkar, A. V., & Feng, Y. (2023). L1-penalized Multinomial Regression: Estimation, inference, and prediction, with an application to risk factor identification for different dementia subtypes. arXiv preprint arXiv:2302.02310.
#'
#' Van de Geer, S., Bühlmann, P., Ritov, Y. A., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. The Annals of Statistics, 42(3), 1166-1202.
#'
#' Zhang, C. H., & Zhang, S. S. (2014). Confidence intervals for low dimensional parameters in high dimensional linear models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(1), 217-242.
#'
#' @examples
#' \donttest{
#' # generate data from Model 1 in Tian et al. (2023) with n = 100 and p = 50
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' n <- 50
#' p <- 50
#' K <- 3
#'
#' Sigma <- outer(1:p, 1:p, function(x,y) {
#'   0.9^(abs(x-y))
#' })
#' R <- chol(Sigma)
#' s <- 3
#' beta_coef <- matrix(0, nrow = p+1, ncol = K-1)
#' beta_coef[1+1:s, 1] <- c(1.5, 1.5, 1.5)
#' beta_coef[1+1:s+s, 2] <- c(1.5, 1.5, 1.5)
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


debiased_lasso <- function(x, y, ref = NULL, beta, nfolds = 5, ncores = 1, nlambda = 50, max_iter = 200, tol = 1e-3,
                           lambda.choice = "lambda.min", alpha = 0.05) {
  K <- length(unique(y))

  y.ori <- y
  y <- numeric(length(y.ori))
  if (is.null(ref)) {
    ref <- setdiff(unique(y.ori), colnames(beta))
    message(paste("No reference level is specified! Class '", ref, "' is automatically detected from beta input as the reference class.", sep = ""))
  }

  y.value <- sort(unique(y.ori[y.ori != ref]))
  for (k in 1:(K-1)) {
    y[y.ori == y.value[k]] <- k
  }
  y[y.ori == ref] <- K


  Theta_list <- inv_cov_calc(x = x, y = y, beta = beta, nfolds = nfolds, nlambda = nlambda, max_iter = max_iter,
                             tol = tol, ncores = ncores, lambda.choice = lambda.choice)
  Theta <- Theta_list$Theta


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
  n <- nrow(x)
  pb <- pb_calc(X = x, beta = beta)
  epsilon <- sapply(1:(K-1), function(k){
    y.dummy[, k] - pb[, k]
  })
  Xtepsilon <- as.numeric(sapply(1:(K-1), function(k){
    t(x) %*% epsilon[, k]
  }))
  b <- as.numeric(beta[-1, ]) + Theta %*% Xtepsilon/n
  b[Theta_list$const.index] <- 0


  cov.hat <- matrix(nrow = (K-1)*p, ncol=(K-1)*p)
  for (k1 in 1:(K-1)) {
    for (k2 in 1:(K-1)) {
      if (k1 == k2) {
        cov.hat[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x) %*% diag(pb[,k1]*(1-pb[,k1])) %*% x)/n
      } else {
        cov.hat[(1+(k1-1)*p):(k1*p), (1+(k2-1)*p):(k2*p)] <- (t(x) %*% diag(-pb[,k1]*pb[,k2]) %*% x)/n
      }
    }
  }


  sigma2 <- sapply(1:p.big, function(j){
    t(Theta[j, ]) %*% cov.hat %*% Theta[j, ]
  })
  sigma2[Theta_list$const.index] <- Inf


  test.stats.value <- sqrt(n)*b/sqrt(sigma2)
  CI <- data.frame(beta = b, p_value = 2*pnorm(-abs(test.stats.value)), CI_lower = b-qnorm(1-alpha/2)*sqrt(sigma2)/sqrt(n), CI_upper = b+qnorm(1-alpha/2)*sqrt(sigma2)/sqrt(n), std_dev = sqrt(sigma2)/sqrt(n))
  CI$std_dev[Theta_list$const.index] <- NA
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
