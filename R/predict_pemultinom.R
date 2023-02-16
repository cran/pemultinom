#' Make predictions on new predictors based on fitted l1-penalized multinomial regression model.
#'
#' Make predictions on new predictors based on fitted l1-penalized multinomial regression model, by using the fitted beta.
#' @export
#' @param beta the beta estimate from l1-penalized multinomial regression. Should be in the same format as \code{beta.min} or \code{beta.1se} in output of function \code{\link{cv.pemultinom}}. The user is recommended to directly pass \code{beta.min} or \code{beta.1se} from the output of function \code{\link{cv.pemultinom}} to parameter \code{beta}.
#' @param ref the reference level, which should be the same reference level as used in obtaining \code{beta}. An input is required.
#' @param xnew new observations to predict labels for. Should be a matrix or a data frame, where each row and column represents an observation and predictor, respectively.
#' @param type the type of prediction output. Can be 'class' or 'prob'. Default = 'class'.
#' \itemize{
#' \item class: the predicted class/label.
#' \item prob: the predicted probability for each class.
#' }
#' @return When \code{type} = 'class', return a vector. When \code{type} = 'prob', return a matrix where each row and column represent an observation and a probability of that class, respectively. Default = 'class'.
#' @seealso \code{\link{cv.pemultinom}}, \code{\link{debiased_lasso}}.
#' @references
#' Tian, Y., Rusinek, H., Masurkar, A. V., & Feng, Y. (2023). L1-penalized Multinomial Regression: Estimation, inference, and prediction, with an application to risk factor identification for different dementia subtypes. arXiv preprint arXiv:2302.02310.
#'
#' @examples
#' # generate training data from Model 1 in Tian et al. (2023) with n = 50 and p = 50
#' set.seed(1, kind = "L'Ecuyer-CMRG")
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

predict_pemultinom <- function(beta, ref, xnew, type = c("class", "prob")) {
  label.name <- c(colnames(beta), ref)
  type <- match.arg(type)
  exp_eta <- sapply(1:ncol(beta), function(l){
    exp(xnew %*% beta[-1, l] + beta[1, l])
  })
  exp_eta <- cbind(exp_eta, 1)
  prob <- exp_eta/rowSums(exp_eta)
  colnames(prob) <- 1:ncol(exp_eta)
  if (type == "prob") {
    return(prob)
  } else if (type == "class"){
    y_pred <- apply(prob, 1, which.max)
    for (k in 1:length(label.name)) {
      y_pred[y_pred == k] <- label.name[k]
    }
    return(y_pred)
  }
}
