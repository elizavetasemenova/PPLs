data{
  int<lower=1> K;                     // number of classes
  int<lower=1> N;                     // number of observations
  int<lower=1> D;                     // number of predictors
  matrix[N, D] X;                     // design matrix
  int<lower=1, upper=K> y[N];         // observed outcome
}

parameters{
  vector[D] z;
  real<lower=0> sigma;
  ordered[K-1] cutpoints;  
}

transformed parameters{
  vector[D] beta = sigma * z;
  vector[N] lp = X * beta;
}

model{
  
  // priors
  z ~ normal(0, 1);
  sigma ~ normal(0, 1);
  
  cutpoints ~ normal(0, 20);
  
  // likelihood
  for (i in 1:N){
   y[i] ~ ordered_logistic(lp[i], cutpoints);  
  }
  
}

generated quantities{
  vector[N] log_lik;        
  real y_pred[N];
  
  for (i in 1:N){

    //log-likelihood
    log_lik[i] = ordered_logistic_lpmf(y[i] | lp[i], cutpoints);
  
     //predictions
     y_pred[i] = ordered_logistic_rng(lp[i], cutpoints);
  }
}
