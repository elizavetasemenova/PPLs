#------------------------------
#------------------------------
# R / Stan - ordinal regression
#------------------------------
#------------------------------

# clear space
rm(list=ls())

# load libraries
library(rstan)
library(carData)
library(caret)

#------------------------------
# Prepare data
#------------------------------
data(WVS)
df = WVS

# outcome
y <- as.numeric(df$poverty)

# predictors
X <- model.matrix(~ 0 + religion + degree + country + gender, data = df)
head(X)

# add age to predictors
age_sc <- (df$age - mean(df$age) ) / sd(df$age)
X <- cbind(X, age_sc)

df_out <- cbind(X, y)
write.csv(df_out, file = "WVS.csv")

K = length(table(y))
N = nrow(X)
D = ncol(X) 

dat =  list(K=K, N=N, D=D, X=X, y=y)

#------------------------------
# Model specification
#------------------------------
mod <- stan_model("3_ordered_logistic_regression.stan")

#------------------------------
# Sampling
#------------------------------
niter = 1000
nchains = 3
thin = 1

m <- sampling(mod, 
              data=dat, 
              iter=niter, 
              chains=nchains, 
              control = list(adapt_delta = 0.99),
              seed=123,
              thin=thin,
              cores=nchains)

#------------------------------
# Parameter estimates
#------------------------------
fit_summary <- summary(m, pars =c("beta", "cutpoints") ,probs = c(0.025, 0.975))
print(fit_summary$summary)

#------------------------------
# Predictions
#------------------------------
y_pred_samps <- rstan::extract(m, pars ="y_pred")[[1]]
str(y_pred_samps)

y_pred = rep(NA, nrow(y_pred_samps))

for (i in 1:ncol(y_pred_samps)){
  probs = c(mean(y_pred_samps[,i] == 1), mean(y_pred_samps[,i] == 2), mean(y_pred_samps[,i] == 3))
  #print(probs)
  y_pred[i] = max(which(probs == max(probs)))
}

table(y_pred)

#------------------------------
# Accuracy, confusion matrix
#------------------------------
(acc = mean(y == y_pred))

confusionMatrix(data=factor(y_pred), reference=factor(y))$ table
