#include <Rcpp.h>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//


// [[Rcpp::export]]
double vec_prod(const NumericVector& x, const NumericVector& y) {
  return std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
}
// double vec_prod(NumericVector x, NumericVector y) {
//   int n = x.size();
//   double total = 0;
//   for (int i = 0; i < n; ++i) {
//     total += x[i]*y[i];
//   }
//   return total;
// }


// [[Rcpp::export]]
NumericVector vec_diff(const NumericVector& x, const NumericVector& y) {
  NumericVector result(x.size());
  std::transform(x.begin(), x.end(), y.begin(), result.begin(), std::minus<double>());
  return result;
}
// NumericVector vec_diff(NumericVector x, NumericVector y) {
//   int n = x.size();
//   NumericVector vec(n);
//   for (int i = 0; i < n; ++i) {
//     vec[i] = x[i]-y[i];
//   }
//   return vec;
// }


// [[Rcpp::export]] // the product of the i-th row of x and y
double mx_vec_prod_i(NumericMatrix x, NumericVector y, int i) {
  int p = y.size();
  double total = 0;
  for (int j = 0; j < p; ++j) {
    total += x(i,j)*y[j];
  }
  return total;
}


// [[Rcpp::export]]
NumericMatrix mx_prod(NumericMatrix x, NumericMatrix y) {
  int nrow = x.nrow(), ncol = y.ncol(), ninter = x.ncol();
  NumericMatrix pd(nrow, ncol);
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      for (int k = 0; k < ninter; ++k) {
        pd(i,j) += x(i,k)*y(k,j);
      }
    }
  }
  return pd;
}


// [[Rcpp::export]]
NumericMatrix xb_mx_plus(NumericMatrix xb, NumericVector beta0) {
  int nrow = xb.nrow(), ncol = xb.ncol();
  NumericMatrix xb_new(nrow, ncol);
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      xb_new(i,j) += beta0[j];
    }
  }
  return xb_new;
}


// [[Rcpp::export]]
double vec_sum(const NumericVector& x) {
  return std::accumulate(x.begin(), x.end(), 0.0);
}
// double vec_sum(NumericVector x) {
//   double total = 0;
//   for (int i = 0; i < x.size(); i++) {
//     total += x[i];
//   }
//   return total;
// }


// [[Rcpp::export]]
double max2(double a, double b) {
  return std::max(a, b);
}
// double max2(double x, double y) {
//   if (x > y) {
//     return x;
//   } else {
//     return y;
//   }
// }


// [[Rcpp::export]]
double vec_max(NumericVector x) {
  double x_max = x[0];
  for (int i = 0; i < x.size()-1; i++) {
    if (x[i+1] > x[i]) {
      x_max = x[i+1];
    }
  }
  return x_max;
}



// [[Rcpp::export]]
NumericVector vec_devide_number(NumericVector x, double y) {
  NumericVector vec(x.size());
  for (int i = 0; i < x.size(); i++) {
    vec[i] = x[i]/y;
  }
  return vec;
}



// [[Rcpp::export]]
NumericVector vec_multiply_number(NumericVector x, double y) {
  NumericVector vec(x.size());
  for (int i = 0; i < x.size(); i++) {
    vec[i] = x[i]*y;
  }
  return vec;
}


// [[Rcpp::export]]
NumericVector vec_square(NumericVector x) {
  NumericVector x_sq(x.size());
  for (int i = 0; i < x.size(); i++) {
    x_sq[i] = pow(x[i], 2);
  }
  return x_sq;
}


// [[Rcpp::export]]
NumericVector vec_pointwise_prod(NumericVector x, NumericVector y) {
  NumericVector xy(x.size());
  for (int i = 0; i < x.size(); i++) {
    xy[i] = x[i]*y[i];
  }
  return xy;
}

// [[Rcpp::export]]
NumericMatrix pr(const NumericMatrix& xb) {
  int n = xb.nrow(), k = xb.ncol();
  NumericMatrix prob(n, k + 1);
  for (int i = 0; i < n; ++i) {
    double sum_exp = 1.0;
    for (int l = 0; l < k; ++l) {
      sum_exp += std::exp(xb(i, l));
    }
    for (int l = 0; l < k; ++l) {
      prob(i, l) = std::exp(xb(i, l)) / sum_exp;
    }
    prob(i, k) = 1.0 / sum_exp;
  }
  return prob;
}

// NumericMatrix pr(NumericMatrix xb) {
//   int nrow = xb.nrow(), ncol = xb.ncol();
//   double pb_sum = 0;
//   NumericMatrix prob(nrow, ncol+1);
//   NumericMatrix exp_xb(nrow, ncol+1);
//   for (int i = 0; i < nrow; ++i) {
//     for (int l = 0; l < ncol+1; ++l) {
//       if (l != ncol) {
//         exp_xb(i,l) = exp(xb(i,l));
//       } else {
//         exp_xb(i,l) = 1;
//       }
//     }
//   }
//
//   for (int i = 0; i < nrow; ++i) {
//     pb_sum = vec_sum(exp_xb(i,_));
//     for (int l = 0; l < ncol+1; ++l) {
//       prob(i,l) = exp_xb(i,l)/pb_sum;
//     }
//   }
//
//   // return exp_xb;
//   return prob;
// }


// [[Rcpp::export]]
NumericMatrix xb_calc(NumericMatrix x, NumericMatrix beta, NumericVector beta0) {
  NumericMatrix xb(x.nrow(), beta.ncol());
  xb = mx_prod(x, beta);
  return xb_mx_plus(xb, beta0);
}


// [[Rcpp::export]]
double abs_value(double x) {
  return std::abs(x);
}
// double abs_value(double x) {
//   if(x >= 0) {
//     return x;
//   } else {
//     return -x;
//   }
// }


// [[Rcpp::export]]
double vec_max_norm(const NumericVector& x) {
  return *std::max_element(x.begin(), x.end(), [](double a, double b) { return std::abs(a) < std::abs(b); });
}
// double vec_max_norm(NumericVector x) {
//   double x_max = abs_value(x[0]);
//   for (int i = 0; i < x.size()-1; i++) {
//     if (abs_value(x[i+1]) > abs_value(x[i])) {
//       x_max = abs_value(x[i+1]);
//     }
//   }
//   return x_max;
// }


// [[Rcpp::export]]
NumericMatrix test(NumericMatrix x) {
  return x(Range(0, 1), Range(0,2));
}

// [[Rcpp::export]]
double soft_thresholding(double z, double lambda) {
  if (z > lambda) return z - lambda;
  if (z < -lambda) return z + lambda;
  return 0;
}
// double soft_thresholding(double z, double gamma) {
//   if (z > 0 && gamma < abs_value(z)) {
//     return z-gamma;
//   } else if (z < 0 && gamma < abs_value(z)) {
//     return z + gamma;
//   } else {
//     return 0;
//   }
// }


// [[Rcpp::export]]
List pemultinom_c(NumericMatrix x, NumericMatrix y, NumericVector lambda_list, int max_iter, double tol) {
  int n = x.nrow(), p = x.ncol(), count = 0, count_l = 0, K = y.ncol(), nlambda = lambda_list.size();
  double change = 100, change_l = 10, v_j = 0, z_j = 0, beta0_l = 0, beta_l = 0, shift_beta0_l = 0, beta0_l_old = 0, max_change = 0, lambda = 0;
  NumericVector w_l(n), z_l(n), x_j(n), r_l(n), beta0(K-1), s_l(n), shift_beta_l(p), beta_l_old(p);
  NumericMatrix prob(n, K), w(n, K-1), z(n, K-1), xb(n, K), beta(p, K-1), beta_full(p*nlambda, K-1), beta0_full(nlambda, K-1);

  for (int r = 0; r < nlambda; r++) {
    // Rcout << "Fitting the model on " << r << "-th lambda value" << "\n";
    lambda = lambda_list[r];
    if (r == 0) {
      xb = xb_mx_plus(mx_prod(x, beta), beta0);
    }
    change = 100;
    count = 0;

    while (count < max_iter && change > tol) {
      change = 0;

      count = count + 1;
      for (int l = 0; l < K-1; ++l) {
        beta_l_old = beta(_, l);
        beta0_l_old = beta0[l];

        count_l = 0;
        change_l = 10;

        while (count_l < max_iter && change_l > tol) {
          change_l = 0;

          count_l = count_l + 1;
          prob = pr(xb);
          for (int i = 0; i < n; ++i) {
            w_l[i] = max2(prob(i, l)*(1-prob(i, l)), 0.0001);
            s_l[i] = y(i, l) - prob(i, l);
            r_l[i] = s_l[i]/w_l[i];
          }

          // Rcout << beta0[l] << "\n";

          beta0_l = beta0[l];
          beta0[l] = beta0[l] + vec_sum(s_l)/vec_sum(w_l);
          shift_beta0_l = beta0[l]-beta0_l;
          if (change_l < abs_value(shift_beta0_l)) {
            change_l = abs_value(shift_beta0_l);
          }
          if (shift_beta0_l != 0) {
            for (int i = 0; i < n; ++i) {
              r_l[i] = r_l[i] - shift_beta0_l;
            }
          }

          for (int j = 0; j < p; ++j) {
            x_j = x(_, j);
            v_j = vec_prod(x_j, vec_pointwise_prod(x_j, w_l))/n;
            z_j = vec_prod(x_j, vec_pointwise_prod(r_l, w_l))/n + v_j*beta(j, l);
            // z_j = vec_prod(x_j, s_l)/n + v_j*beta(j, l);
            beta_l = beta(j, l);
            beta(j, l) = soft_thresholding(z_j, lambda)/v_j;
            shift_beta_l[j] = beta(j, l)-beta_l;
            if (change_l < abs_value(shift_beta_l[j])) {
              change_l = abs_value(shift_beta_l[j]);
            }

            if (shift_beta_l[j] != 0) {
              for (int i = 0; i < n; ++i) {
                r_l[i] = r_l[i] - shift_beta_l[j]*x(i, j);
              }
            }

          }

          for (int i = 0; i < n; ++i) {
            xb(i, l) = xb(i, l) + vec_prod(x(i,_), shift_beta_l) + shift_beta0_l;
          }

          // Rcout << w_l[0] << "\n";
        }
        // if (r > 0) {
        //   Rcout << count << "\n";
        // }

        max_change = max2(vec_max_norm(vec_diff(beta(_, l), beta_l_old)), abs_value(beta0[l]-beta0_l_old));
        if (change < max_change) {
          change = max_change;
        }
      }
    }

    for (int l = 0; l < K-1; l++) {
      for (int j = 0; j < p; j++) {
        beta_full(r*p+j, l) = beta(j, l);
      }
      beta0_full(r, l) = beta0[l];
    }
  }

  return List::create(beta_full, beta0_full);
}


// [[Rcpp::export]]
NumericMatrix penalized_quad(NumericMatrix A, NumericVector b, NumericVector lambda_list, int max_iter, double tol) {
  int p = A.ncol(), count = 0, nlambda = lambda_list.size();
  double change = 100, lambda = 0;
  NumericVector gamma(p), z(p), gamma_old(p);
  NumericMatrix gamma_full(nlambda, p);
  for (int r = nlambda-1; r >= 0; r--) {
    change = 100;
    lambda = lambda_list[r];
    count = 0;

    while (count < max_iter && change > tol) {
      for (int j = 0; j < p; j++) {
        if (A(j,j) == 0) {
          gamma_old[j] = gamma[j];
          gamma[j] = 0;
        } else {
          gamma_old[j] = gamma[j];
          z[j] = b[j] - vec_prod(A(j,_), gamma) + A(j, j)*gamma[j];
          gamma[j] = soft_thresholding(z[j], lambda)/A(j, j);
        }

      }
      change = vec_max_norm(vec_diff(gamma, gamma_old));
      count = count + 1;
    }

    for (int j = 0; j < p; j++) {
      gamma_full(r, j) = gamma[j];
    }
  }
  return gamma_full;
}




// [[Rcpp::export]]
List pemultinom_c_reverse(NumericMatrix x, NumericMatrix y, NumericVector lambda_list, int max_iter, double tol, NumericVector zero_ind, NumericVector weights, bool intercept) {
  int n = x.nrow(), p = x.ncol(), count = 0, count_l = 0, K = y.ncol(), nlambda = lambda_list.size();
  double change = 100, change_l = 10, v_j = 0, z_j = 0, beta0_l = 0, beta_l = 0, shift_beta0_l = 0, beta0_l_old = 0, max_change = 0, lambda = 0;
  NumericVector w_l(n), w_l1(n), z_l(n), x_j(n), r_l(n), wr_l(n), beta0(K-1), s_l(n), shift_beta_l(p), beta_l_old(p);
  NumericMatrix prob(n, K), w(n, K-1), z(n, K-1), xb(n, K), beta(p, K-1), beta_full(p*nlambda, K-1), beta0_full(nlambda, K-1), active_set(p, K-1);

  for (int r = nlambda-1; r >= 0; r--) {

    lambda = lambda_list[r];
    if (r == nlambda-1) {
      xb = xb_mx_plus(mx_prod(x, beta), beta0);
      change = 100;
      count = 0;

      while (count < max_iter && change > tol) {
        change = 0;

        count = count + 1;
        for (int l = 0; l < K-1; ++l) {
          beta_l_old = beta(_, l);
          beta0_l_old = beta0[l];

          count_l = 0;
          change_l = 10;

          while (count_l < max_iter && change_l > tol) {

            change_l = 0;

            count_l = count_l + 1;
            prob = pr(xb);
            for (int i = 0; i < n; ++i) {
              w_l1[i] = max2(prob(i, l)*(1-prob(i, l)), 0.0001);
              w_l[i] = weights[i]*w_l1[i];
              s_l[i] = y(i, l) - prob(i, l);
              r_l[i] = s_l[i]/w_l1[i];
              wr_l[i] = weights[i]*s_l[i];
            }

            // Rcout << beta0[l] << "\n";

            beta0_l = beta0[l];
            if (intercept) {
              beta0[l] = beta0[l] + vec_sum(wr_l)/vec_sum(w_l);
            }
            shift_beta0_l = beta0[l]-beta0_l;
            if (change_l < abs_value(shift_beta0_l)) {
              change_l = abs_value(shift_beta0_l);
            }
            if (shift_beta0_l != 0) {
              for (int i = 0; i < n; ++i) {
                r_l[i] = r_l[i] - shift_beta0_l;
              }
            }

            for (int j = 0; j < p; ++j) {
              if (zero_ind[j] == 1) {
                beta(j, l) = 0;
                shift_beta_l[j] = 0;
                continue;
              }
              x_j = x(_, j);
              v_j = vec_prod(x_j, vec_pointwise_prod(x_j, w_l))/n;
              z_j = vec_prod(x_j, vec_pointwise_prod(r_l, w_l))/n + v_j*beta(j, l);
              // z_j = vec_prod(x_j, s_l)/n + v_j*beta(j, l);
              beta_l = beta(j, l);
              beta(j, l) = soft_thresholding(z_j, lambda)/v_j;
              shift_beta_l[j] = beta(j, l)-beta_l;
              if (change_l < abs_value(shift_beta_l[j])) {
                change_l = abs_value(shift_beta_l[j]);
              }

              if (shift_beta_l[j] != 0) {
                for (int i = 0; i < n; ++i) {
                  r_l[i] = r_l[i] - shift_beta_l[j]*x(i, j);
                }
              }

            }

            for (int i = 0; i < n; ++i) {
              xb(i, l) = xb(i, l) + vec_prod(x(i,_), shift_beta_l) + shift_beta0_l;
            }

          }

          max_change = max2(vec_max_norm(vec_diff(beta(_, l), beta_l_old)), abs_value(beta0[l]-beta0_l_old));
          if (change < max_change) {
            change = max_change;
          }
        }
      }
      for (int l = 0; l < K-1; ++l) {
        for (int j = 0; j < p; ++j) {
          if (abs_value(beta(j, l)) > 1e-16) {
            active_set(j, l) = 1;
          }
        }
      }
    } else {
      change = 100;
      count = 0;

      while (count < max_iter && change > tol) {
        change = 0;

        count = count + 1;
        for (int l = 0; l < K-1; ++l) {
          beta_l_old = beta(_, l);
          beta0_l_old = beta0[l];

          count_l = 0;
          change_l = 10;

          while (count_l < max_iter && change_l > tol) {
            change_l = 0;

            count_l = count_l + 1;
            prob = pr(xb);
            for (int i = 0; i < n; ++i) {
              w_l1[i] = max2(prob(i, l)*(1-prob(i, l)), 0.0001);
              w_l[i] = weights[i]*w_l1[i];
              s_l[i] = y(i, l) - prob(i, l);
              r_l[i] = s_l[i]/w_l1[i];
              wr_l[i] = weights[i]*s_l[i];
            }

            // Rcout << beta0[l] << "\n";

            beta0_l = beta0[l];
            if (intercept) {
              beta0[l] = beta0[l] + vec_sum(wr_l)/vec_sum(w_l);
            }
            shift_beta0_l = beta0[l]-beta0_l;
            if (change_l < abs_value(shift_beta0_l)) {
              change_l = abs_value(shift_beta0_l);
            }
            if (shift_beta0_l != 0) {
              for (int i = 0; i < n; ++i) {
                r_l[i] = r_l[i] - shift_beta0_l;
              }
            }

            for (int j = 0; j < p; ++j) {
              if (active_set(j, l) == 0) {
                shift_beta_l[j] = 0;
                continue;
              }
              x_j = x(_, j);
              v_j = vec_prod(x_j, vec_pointwise_prod(x_j, w_l))/n;
              z_j = vec_prod(x_j, vec_pointwise_prod(r_l, w_l))/n + v_j*beta(j, l);
              beta_l = beta(j, l);
              beta(j, l) = soft_thresholding(z_j, lambda)/v_j;
              shift_beta_l[j] = beta(j, l)-beta_l;
              if (change_l < abs_value(shift_beta_l[j])) {
                change_l = abs_value(shift_beta_l[j]);
              }

              if (shift_beta_l[j] != 0) {
                for (int i = 0; i < n; ++i) {
                  r_l[i] = r_l[i] - shift_beta_l[j]*x(i, j);
                }
              }

            }

            for (int i = 0; i < n; ++i) {
              xb(i, l) = xb(i, l) + vec_prod(x(i,_), shift_beta_l) + shift_beta0_l;
            }

          }

          max_change = max2(vec_max_norm(vec_diff(beta(_, l), beta_l_old)), abs_value(beta0[l]-beta0_l_old));
          if (change < max_change) {
            change = max_change;
          }
        }
      }
      for (int l = 0; l < K-1; ++l) {
        for (int j = 0; j < p; ++j) {
          if (active_set(j, l) == 0) {
            continue;
          }
          if (abs_value(beta(j, l)) <= 1e-16) {
            active_set(j, l) = 0;
          }
        }
      }
    }


    for (int l = 0; l < K-1; l++) {
      for (int j = 0; j < p; j++) {
        beta_full(r*p+j, l) = beta(j, l);
      }
      beta0_full(r, l) = beta0[l];
    }
  }

  return List::create(beta_full, beta0_full);
}




// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//


