library(tidyverse)
library(glmnet)
library(broom)
library(modelr)
library(parallel)


load("prostate.RData")
prostate <- as.tibble(prostate)

house <- as.tibble(read.csv("house_prices.csv"))
train <- as.tibble(house %>%
                     filter(train == TRUE))
test <- as.tibble(house  %>%
                    filter(train == FALSE))

#### Ridge regression ####
indep <- prostate %>%               # Extracting covariate names
  select(-c(lpsa, group)) %>%
  colnames()
dep <- "lpsa"

ridge <- function(data, dep, indep, lambda){  # Estimating betas
  int <- matrix(rep(1, nrow(data)), ncol = 1) # Intercept ones
  reg <- as.matrix(data %>%           # Indepentent vars
                     select(indep)) 
  X <- cbind(int, reg)                # Regressor matrix
  y <- as.matrix(data %>%             # Dependent variable
                   select(dep))
  lambda <- lambda
  Id <- diag(ncol(X))                 # Identity matrix
  beta_ridge <- solve(t(X)%*%X + lambda*Id) %*% t(X) %*% y
  beta_ridge = as.vector(beta_ridge)  # Vectorizing output
  return(beta_ridge)
}

beta_hat <- ridge(prostate, dep,    # Saving betas
                  indep , 10)
beta_hat
class(beta_hat)                     # Testing class

pred <- function(data, indep, beta_hat){      # Estimating yhats
  int <- matrix(rep(1, nrow(data)), ncol = 1) # Intercept ones
  reg <- as.matrix(data %>%           # Indepentent vars
                     select(indep)) 
  X <- cbind(int, reg)                # Regressor matrix
  B <- as.matrix(beta_hat)            # Betas as matrix
  preds <- X %*% B                    # Predicted values
  preds <- as.vector(preds)
  return(preds)
}

pred_vals <- pred(prostate, indep, beta_hat)
pred_vals <- tibble(Predicted = pred_vals)
pred_vals

cv <- function(data, dep, indep, lambda){
  n <- length(unique(data$group))    # Number of groups
  mse <- numeric(n)                  # Preallocation
  for (i in 1:n){                    # One per group
    X <- data %>%                    # Filtering groups
      filter(group != i)
    X_pred <- data %>%
      filter(group == i)             # Data for prediction
    Y <- X_pred %>% 
      pull(dep)                      # Values to predict
    betas <- ridge(X, dep = dep,     # Dependent var
                   indep = indep,    # Independent vars
                   lambda = lambda)
    preds <- pred(X_pred, indep, 
                  betas)             # Using estimated betas
    mse[i] <- mean((Y - preds)^2)    # Computing MSE
    
  }
  mse <- as.vector(mean(mse))        # Mean of MSE:s
  return(mse)
}


cv(prostate, dep, indep, 10)


l <- seq(0, 50, length.out = 50)
mse <- numeric(length(l))
for(i in 1:length(l)){
  mse[i] <- cv(prostate, "lpsa", indep, l[i])
  
}
which.min(mse)
l
mse %>%
  enframe() %>%
  ggplot() +
  geom_line(aes(x = name, y = value, color = "red")) +
  labs(x = expression(lambda),
       y = "MSE",
       title = expression(MSE~plotted~against~lambda)) +
  theme_classic() +
  theme(plot.title = element_text(face = "bold",              
                                  hjust = 0.5, size = 15),
        axis.title.x = element_text(size = 15),
        panel.grid.major = element_line(color = "gray90")
  ) + 
  guides(color = FALSE)

#### Bagging ####
bag <- function(B, train, test){
  y_hat <- matrix(NA, nrow = 1465, ncol = B)           # Preallocation
  
  for(i in 1:B){
    sample <- sample_n(train, 1465, replace = TRUE) # Sample from sample
    mod <- lm(SalePrice ~ ., data = sample)         # Full model
    mod_sel <- step(mod, trace = FALSE)             # Model selection
    mod_step <- lm(formula(mod_sel), data = sample) # Reduced model
    
    options(warn = -1)                                  # Kills warining message
    y_hat[, i] <- test %>%
      add_predictions(mod_step)    %>%                  # Predictions
      pull(pred)
    options(warn = 1)
  }
  y <- test %>%
    pull(SalePrice)
  hat <- rowMeans(y_hat)
  rmse <- sqrt(mean((y-hat)^2))
  return(rmse)
}


cl <- makeCluster(4)
clusterEvalQ(cl, {
  library(tidyverse)
  library(modelr)
})
clusterExport(cl, varlist = c("train", "test"))

parallel <- parSapply(cl, rep(250,4), FUN = bag, 
                      train = train, test = test)  # Parallel computing
parallel <- mean(parallel)                         # Final value
parallel
stopCluster(cl)

#### Feature Bagging ####
f_bag <- function(B, train, test){
  y_hat <- matrix(NA, nrow = 1465, ncol = B) # Preallocation  
  covariates <- train %>%
    select(-c(SalePrice, train)) %>%
    colnames()                              # Covariates to sample from
  dep <- train %>%                   
    select(SalePrice) %>%
    colnames                                # Dependent variable
  sqp <- round(sqrt(length(covariates)))    # Covariate sample size
  
  for(i in 1:B){
    sample <- sample_n(train, 1465, 
                       replace = TRUE)      # Sample
    sample_covariates <- sample(covariates, sqp, replace = FALSE)           # Sample covariates
    formula <- paste(dep, '~', paste(sample_covariates, collapse = ' + ' )) # Regression formula
    mod <- lm(formula, data = sample)       # Regression model
    
    options(warn = -1)                      # Kills warining message
    y_hat[, i] <- test       %>%
      add_predictions(mod)   %>%            # Predictions
      pull(pred)                            # Extracting RMSE
    options(warn = 1)
  }
  y <- test %>%
    pull(dep)                               # Response
  hat <- rowMeans(y_hat)
  rmse <- sqrt(mean((y - hat)^2))
  return(rmse)
}

f_bag(5000, train, test)


#### Stacking ####
n <- nrow(train)
hat <- matrix(NA, nrow = n, ncol = 4)
x <- model.matrix(SalePrice ~ ., train)           # Preparing the training set
x_test <- model.matrix(SalePrice ~ ., test)
y <- train$SalePrice 
cv_las <- cv.glmnet(x, y, alpha = 1)
cv_rid <- cv.glmnet(x, y, alpha = 0)
mod_step <- step(lm(SalePrice ~ ., data = train), trace = FALSE)


for(i in 1:n){
  tr <- train[-i, ]
  te <- train[i, ]
  x_tr <- model.matrix(SalePrice ~ ., tr)           # Preparing the training set
  y_tr <- tr$SalePrice 
  x1 <- model.matrix(SalePrice ~ ., te)
  l_mod <- lm(SalePrice ~ ., data = tr)
  las_mod <- glmnet(x_tr, y_tr, alpha = 1)
  rid_mod <- glmnet(x_tr, y_tr, alpha = 0)
  st_mod <- lm(formula(mod_step), data = tr)
  options(warn = -1)
  hat[i, 1] <- predict(l_mod, te)
  hat[i, 2] <- predict(las_mod, x1, s = cv_las$lambda.min, 
                       type = "response")
  hat[i, 3] <- predict(rid_mod, x1, s = cv_rid$lambda.min, 
                       type = "response")
  hat[i, 4] <- predict(st_mod, te)
  options(warn = 1)}

reg_y <- lm(y ~ hat)
b <- matrix(reg_y$coefficients, ncol = 1)

options(warn = -1)
test_lm <- matrix(predict(lm(SalePrice ~ ., data = train), test), ncol = 1)
test_step <- matrix(predict(mod_step, test), ncol = 1)
test_lass <- matrix(predict(glmnet(x_tr, y_tr, alpha = 1), x_test, s = cv_las$lambda.min, 
                            type = "response"), ncol = 1)
test_rid <- matrix(predict(glmnet(x_tr, y_tr, alpha = 1), x_test, s = cv_rid$lambda.min, 
                           type = "response"), ncol = 1)
options(warn = 1)

intercept <- matrix(1, nrow = 1465)

predictions <- as.data.frame(cbind(intercept, test_lm, test_lass, test_rid, test_step))
stack_pred <- predictions %*% b
y_test <- test %>%
  pull(SalePrice)
stack <- tibble(response = y_test,
                predictions = as.numeric(stack_pred))

stack <- stack %>%
  mutate(dev_square = (response - predictions)^2) %>%
  summarize(RMSE = sqrt(mean(dev_square)))
stack
models_rmse[1, 7] <- stack




\documentclass{article}

\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage[a4paper, total={6.1in, 8.1in}]{geometry}

\author{Vilgot \"{O}sterlund}
\title{Solutions to HWA 1}
\date{November 19, 2018}

\begin{document}

\maketitle

\section{}
\textit{Y} belongs to the exponential dispersion family if the probability density or mass function of \textit{Y} is of the form:


\begin{align*}
f(y_i; \theta_i) = exp \left\{\frac{y_i \theta_i - b(\theta_i)}{a(\phi)} + c(y_i, \phi) \right\}, 
\end{align*}

\noindent Where $\theta_i$ is the natural parameter and $\phi$ is the dispersion parameter, $\phi > 0$.

\noindent We have:

\begin{align*}
f(y_i; \theta_i) =\sqrt{\frac{1}{2 \pi y^{3}_{i}}}exp\left\{\frac{\theta_i\left(y_i - \sqrt{\frac{1}{2 \theta_i}} \right)^2}{y_i} \right\}.
\end{align*}


\begin{align*}
f(y_i; \theta_i) &=\sqrt{\frac{1}{2 \pi y^{3}_{i}}}exp\left\{\frac{\theta_i\left(y_i - \sqrt{\frac{1}{2 \theta_i}} \right)^2}{y_i} \right\} \\ 
&=exp\left\{log(2 \pi y^{3}_{i})^\frac{-1}{2} \right\} exp\left\{\frac{\theta_i\left(y_i - \sqrt{\frac{1}{2 \theta_i}} \right)^2}{y_i} \right\} \\
&=exp\left\{\frac{\theta_i\left(y^{2}_{i} - 2y_i\sqrt{\frac{-1}{2\theta_i}} + \sqrt{\frac{-1}{2\theta_i}}^2 \right)}{y_i} -\frac{log(2 \pi y^{3}_{i})}{2} \right\} \\
&=exp\left\{\theta_i y_i - 2 \theta_i \sqrt{\frac{-1}{2\theta_i}} + \frac{\frac{\theta_i}{2\theta_i}}{y_i} - \frac{log(2 \pi y^{3}_{i})}{2} \right\} \\
&=exp\left\{\theta_i y_i - 2 \theta_i \sqrt{\frac{-1}{2\theta_i}} + \frac{\frac{1}{2}}{y_i} - \frac{log(2 \pi y^{3}_{i})}{2} \right\} \\
&=exp\left\{\underbrace{y_i \theta_i - 2 \theta_i \sqrt{\frac{-1}{2\theta_i}}}_\frac{y_i \theta_i - b(\theta_i)}{a(\phi)} + \underbrace{\left[ \frac{1}{2 y_i} - \frac{log(2 \pi y^{3}_{i})}{2} \right] }_{c(y_i)} \right\}
\end{align*}

\noindent Hence, $\theta_i$ is the natural parameter and $a(\phi) = 1$.

\section{}

Since $Y_i$ belongs to the exponential family and is a continous random variable, we have differentiation that can pass trough integration. This gives that $E(Y_i) = b'(\theta_i)$. We can rewrite our $b(\theta_i)$ to: 

\begin{align*}
2 \theta_i \sqrt{\frac{-1}{2\theta_i}} &= - 2 \theta_i \sqrt{- 2 \theta^{-1}} \\
&= - 2 \theta_i (- 2 \theta^{-1})^{\frac{1}{2}} \\
&= - 2 \theta_i^{\frac{1}{2}} \\
&= \sqrt{- 2\theta_i}
\end{align*}

\noindent With the help of the chain rule, $y' = f'(g(x))g'(x)$, letting $f(g(x)) = \sqrt{- 2\theta_i}$ and $g(x) = - 2 \theta_i$, we have that:

\begin{align*}
b`(\theta_i) &= \frac{1}{2}\theta_i^{-\frac{1}{2}}(- 2) \\
&= \frac{1}{\sqrt{-2\theta_i}}
\end{align*}

\noindent Hence, the mean of $Y_i$ is $\frac{1}{\sqrt{-2\theta_i}}$. The variance of $Y_i$ is given by the second order derivative of $b(\theta_i)$, which is the first order derivative of $\frac{1}{\sqrt{-2\theta_i}}$. Again, using the chain rule:

\begin{align*}
b``(\theta_i) &= \frac{1}{(-2\theta_i)^{\frac{3}{2}}}
\end{align*}

Hence, the variation of $Y_i$ is $\frac{1}{(-2\theta_i)^{\frac{3}{2}}}$.

\section{}

The canonial link function is:

\begin{align*}
\mu &= \frac{1}{\sqrt{-2\theta_i}} \\
\sqrt{-2\theta_i} &= \frac{1}{\mu} \\
2\theta_i &= \frac{1}{\mu^2} \\
\theta_i &= \frac{1}{2\mu^2}
\end{align*}

\section{}

We have that $log(\mu_i) = \eta_i = \beta_1 + \beta_2X_{i1} + \cdots + \beta_p X_i$. The expression for the IRLS is $\beta^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} Z^{(t)}$, where $W^{(t)} = D^{(t)}(V^{(t)})^{-1}D^{(t)}$ and $Z^{(t)} = X\beta^t + (D^{(t)})^{-1} (y - \mu^t)$. 

\[ X = \left( \begin{array}{ccc}
x_{11} &\cdots & x_{1p} \\
\vdots & \ddots & \vdots \\
x_{n1} & \cdots & x_{np} \end{array} \right),\] 

\[ D = \left( \begin{array}{ccc}
exp(\beta_1 + \beta_2X_{i1} + \cdots + \beta_p X_i) &\cdots & x_{1n} \\
\vdots & \ddots & \vdots \\
0 & \cdots & exp(\beta_1 +\beta_2X_{i1} + \cdots + \beta_p X_i) \end{array} \right),\] 

\[ V = \left( \begin{array}{ccc}
exp(\beta_1 + \beta_2X_{i1} + \cdots + \beta_p X_i) &\cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & exp(\beta_1 +\beta_2X_{i1} + \cdots + \beta_p X_i) \end{array} \right),\] 


\[ Y - \mu^{(t)} = \left( \begin{array}{ccc}
y_i -  exp(\beta_1 + \beta_2X_{i1} + \cdots + \beta_p X_i)\\
\vdots\\
y_i -  exp(\beta_1 + \beta_2X_{i1} + \cdots + \beta_p X_i) \end{array} \right),\]

\section{}
For the saturated model, we have that every $Y_i$ has it own $\mu_i$. The maximum likelihood estimate (MLE) of $\mu_i$ is $\tilde{\mu}_i$ and the MLE of $\theta_i$ is $\tilde{\theta}_i = \frac{1}{2\mu^2}$.
\section{}

\section{}
\section{}
\section{}
\section{}
\end{document}
