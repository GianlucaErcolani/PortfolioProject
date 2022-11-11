## Data Import and Exploration
df_train=read.csv("C:/Users/gianl/Downloads/house-prices-advanced-regression-techniques (1)/train.csv")

names(df_train)

dim(df_train)

## Handling missing values. We can delete the variables or impute some values. In this case we decide to delete, because I think these variables are negligible and have a lot of missing values. Like 'PoolQC' (Pool Quality) and MiscFeature (Miscellaneous feature not covered in other categories).
df_train = subset(df_train, select=-c(PoolQC, MiscFeature, Alley, Fence, FireplaceQu, LotFrontage, GarageCond, GarageType, GarageYrBlt, GarageFinish, GarageQual, BsmtExposure, BsmtFinType2, BsmtFinType1, BsmtCond, BsmtQual, MasVnrArea, MasVnrType))


## We have one missing observation in 'Electrical'. Since it is just one observation, we'll delete this observation and keep the variable.
library(tidyr)
df_train = df_train %>% drop_na(Electrical)

summary(df_train)

dim(df_train)

sum(is.na(df_train))

## Now add factors for variables that are factors

df_train$MSZoning <- as.factor(df_train$MSZoning)
df_train$Street <- as.factor(df_train$Street)
df_train$LotShape <- as.factor(df_train$LotShape)
df_train$LandContour <- as.factor(df_train$LandContour)
df_train$Utilities <- as.factor(df_train$Utilities)
df_train$LotConfig <- as.factor(df_train$LotConfig)
df_train$LandSlope <- as.factor(df_train$LandSlope)
df_train$Neighborhood <- as.factor(df_train$Neighborhood)
df_train$Condition1 <- as.factor(df_train$Condition1)
df_train$Condition2 <- as.factor(df_train$Condition2)
df_train$BldgType <- as.factor(df_train$BldgType)
df_train$HouseStyle <- as.factor(df_train$HouseStyle)
df_train$RoofStyle <- as.factor(df_train$RoofStyle)
df_train$RoofMatl <- as.factor(df_train$RoofMatl)
df_train$Exterior1st <- as.factor(df_train$Exterior1st)
df_train$Exterior2nd <- as.factor(df_train$Exterior2nd)
df_train$ExterQual <- as.factor(df_train$ExterQual)
df_train$ExterCond <- as.factor(df_train$ExterCond)
df_train$Foundation <- as.factor(df_train$Foundation)
df_train$Heating <- as.factor(df_train$Heating)
df_train$HeatingQC <- as.factor(df_train$HeatingQC)
df_train$CentralAir <- as.factor(df_train$CentralAir)
df_train$Electrical <- as.factor(df_train$Electrical)
df_train$KitchenQual <- as.factor(df_train$KitchenQual)
df_train$Functional <- as.factor(df_train$Functional)
df_train$PavedDrive <- as.factor(df_train$PavedDrive)
df_train$SaleType <- as.factor(df_train$SaleType)
df_train$SaleCondition <- as.factor(df_train$SaleCondition)
df_train$Exterior2nd <- as.factor(df_train$Exterior2nd)
df_train$ExterQual <- as.factor(df_train$ExterQual)
df_train$ExterCond <- as.factor(df_train$ExterCond)

## Checking for multicollinearity
# As a rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity
# Select numeric variables
num <- unlist(lapply(df_train, is.numeric))  
df_train_num <- df_train[ , num] 
df_train_num

# Correlation matrix (Pearson correlation) of numeric variables

install.packages('ggcorrplot')
library(ggcorrplot)
cor(df_train_num[, colnames(df_train_num)[colnames(df_train_num) != 'SalePrice']] )
cor_matrix = round(cor(df_train_num[, colnames(df_train_num)[colnames(df_train_num) != 'SalePrice']] ), 1)
ggcorrplot(cor_matrix, lab = TRUE)

# Delete variables having correlation greater or equal to 0.8
cor_matrix_rm <- cor_matrix
cor_matrix_rm[upper.tri(cor_matrix_rm)] <- 0
diag(cor_matrix_rm) <- 0
cor_matrix_rm

df_train_num_new <- df_train_num[ , !apply(cor_matrix_rm, 2, function(x) any(x >= 0.8))]


# VIF calculation 
lm.fit <- lm(SalePrice ~ ., data = df_train_num_new)
summary(lm.fit)
library(car)
vif(lm.fit)
# VIF is about 5/6 for 'BsmtFinSF1', 'X1stFlrSF' and 'X2ndFlrSF'. I decide to maintain all the variables.

# Scaling

df_train_num_new = scale(df_train_num_new)

df_train_num_new = data.frame(df_train_num_new)

# Feature selection
# Best subset selection. 'Leaps' library works only for numeric features

library(leaps)
regfit.full <- regsubsets(SalePrice ~ ., data = df_train_num_new)
summary(regfit.full)
reg.summary <- summary(regfit.full)
names(reg.summary)

reg.summary$rsq

## Plot RSS, adjusted R2, Cp, and BIC for all of the models at once

par(mfrow = c(2, 2))
plot(reg.summary$rss , xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg.summary$adjr2 , xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
which.max(reg.summary$adjr2)
points (8, reg.summary$adjr2 [8] , col = "red", cex = 2, pch = 20)
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(reg.summary$cp)
points (8, reg.summary$cp[8] , col = "red", cex = 2, pch = 20)
which.min(reg.summary$bic)
plot(reg.summary$bic , xlab = "Number of Variables", ylab = "BIC", type = "l")
points (8, reg.summary$bic [8], col = "red", cex = 2, pch = 20)


## Display the selected variables for the best model with a given number of predictors, ranked according to the BIC, Cp, adjusted R2, or AIC

plot(regfit.full , scale = "r2")
plot(regfit.full , scale = "adjr2")
plot(regfit.full , scale = "Cp")
plot(regfit.full , scale = "bic")

## Training set RSS and training set R2 cannot be used to select from among a set of models with different numbers of variables
## We can use other metrics:  Mallow’s Cp, Bayesian information criterion (BIC), Akaike information  criterion (AIC), and adjusted R2
## If we use BIC, the best model has 8 variables

coef(regfit.full, 8)

## Linear regression
## Linear regression model has several potential problems: non-linearity of the response-predictor relationships, correlation of error terms, non-constant variance of error terms.
## Let's check these potential problems.

# Linearity 

ggplot(df_train_num_new, aes(SalePrice, MSSubClass))+geom_point()+
        labs(title='Scatter plot of SalePrice and MSSubClass',x='MSSubClass', y='SalePrice')

ggplot(df_train_num_new, aes(SalePrice, OverallQual))+geom_point()+
        labs(title='Scatter plot of SalePrice and OverallQual',x='OverallQual', y='SalePrice')

ggplot(df_train_num_new, aes(SalePrice, OverallCond))+geom_point()+
        labs(title='Scatter plot of SalePrice and OverallCond',x='OverallCond', y='SalePrice')

ggplot(df_train_num_new, aes(SalePrice, YearBuilt))+geom_point()+
        labs(title='Scatter plot of SalePrice and YearBuilt',x='YearBuilt', y='SalePrice')

ggplot(df_train_num_new, aes(SalePrice, BsmtFinSF1))+geom_point()+
        labs(title='Scatter plot of SalePrice and BsmtFinSF1',x='BsmtFinSF1', y='SalePrice')

ggplot(df_train_num_new, aes(SalePrice, X1stFlrSF))+geom_point()+
        labs(title='Scatter plot of SalePrice and X1stFlrSF',x='X1stFlrSF', y='SalePrice')

ggplot(df_train_num_new, aes(SalePrice, X2ndFlrSF))+geom_point()+
        labs(title='Scatter plot of SalePrice and X2ndFlrSF',x='X2ndFlrSF', y='SalePrice')

ggplot(df_train_num_new, aes(SalePrice, GarageArea))+geom_point()+
        labs(title='Scatter plot of SalePrice and GarageArea',x='GarageArea', y='SalePrice')



# Create the new dataset with variables selected by BIC criterion

df_BIC = df_train_num_new[c('SalePrice', 'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtFinSF1', 'X1stFlrSF', 'X2ndFlrSF', 'GarageArea')]


## Features

x = df_BIC[, -1]

## Target

y = df_BIC[, 1]

# Shuffling and splitting
# Use 70% of dataset as training set and 30% as test set

sample <- sample(c(TRUE, FALSE), nrow(df_BIC), replace=TRUE, prob=c(0.7,0.3))
train  <- df_BIC[sample, ]
test   <- df_BIC[!sample, ]

x_train = train[, -1]
x_test = test[, -1]
y_train = train[, 1]
y_test = test[, 1]


## Fitting and predicting

lm.fit <- lm(y_train ~ ., data = x_train)
model_summ = summary(lm.fit)

# Adjusted R-squared 0.77  


data_mod <- data.frame(Predicted = predict(lm.fit, x_test), Observed = y_test)    # Create data for ggplot2


# Draw plot using ggplot2 package

ggplot(data_mod, aes(x = Predicted , y = Observed)) + geom_point() + geom_abline(intercept = 0, slope = 1, color = "red", size = 2)

# Residual plot

data_mod["residuals"] = data_mod[c("Observed")] - data_mod[c("Predicted")]

ggplot(data_mod, aes(x = Predicted , y = residuals)) + geom_point() + labs(title='Residual plot') + geom_hline(yintercept=0, linetype='dashed', col = 'red')

# There is a little pattern in the residuals. It seems like there is slight heteroskedasticity in the data. It seems like there is also little autocorrelation in the left side and in the right side of the plot.

# Residual plots are a useful graphical tool for identifying non-linearity.
# If the true relationship is far from linear, then virtually all of the conclusions that we draw from the fit are suspect. In addition, the prediction accuracy of the model can be significantly reduced.

# If there is correlation among the error terms, then the estimated standard errors will tend to underestimate the true standard errors. As a result, confidence and prediction intervals will be narrower than they should be.

# The standard errors, confidence intervals, and hypothesis tests associated with the linear model rely upon the assumption of non-constant variance of the error terms.


# Measuring performance. 
# MSE is measured in units that are the square of the target variable, while RMSE is measured in the same units as the target variable.
# MSE penalizes larger errors more severely.
# MSE(Mean squared error)

mean((data_mod$Predicted - data_mod$Observed)^2)

# Root mean square error

sqrt(mean((data_mod$Predicted - data_mod$Observed)^2))

# Mean absolute error (MAE): it measures the absolute average distance between the real data and the predicted data, but it fails to punish large errors in prediction.

install.packages('Metrics')
library(Metrics)
mae(data_mod$Observed, data_mod$Predicted)


# Alternative fitting procedures other than least squares that can yield better prediction accuracy and model interpretability.
# Regularization (aka shrinkage)
# The estimated coefficients are shrunken towards zero relative to the least squares estimates.
# It has the effect of reducing variance.
# The two best-known shrinkage methods are: ridge regression and lasso.

# Let's start with ridge regression
# Ridge regression. It is particularly useful to mitigate the problem of multicollinearity in linear regression
# In general, the method provides improved efficiency in parameter estimation problems in exchange for a tolerable amount of bias.

library(glmnet)
x <- model.matrix(y_train ~ ., x_train)
y <- y_train

# By default the glmnet() function performs ridge regression for an automatically selected range of λ values.
# If alpha=0 then a ridge regression model is fit, and if alpha=1 then a lasso model is fit.
# However, here we have chosen to implement the function over a grid of values ranging from λ = 10^10 to λ = 10^-2, essentially covering the full range of scenarios from the null model containing only the intercept, to the least squares fit.
# By default, the glmnet() function standardizes the variables so that they are on the same scale.

grid <- 10^seq(10, -2, length = 100)
ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid)

dim(coef(ridge.mod))

ridge.mod$lambda[50]

coef(ridge.mod)[,50]

# L2 norm. It measures the distance of β from the origin. As λ increases, the L2 norm of β will always decrease.

sqrt(sum(coef(ridge.mod)[-1, 50]^2))

ridge.mod$lambda[60]

coef(ridge.mod)[, 60]

sqrt(sum(coef(ridge.mod)[-1, 60]^2))

# Ridge regression for λ=50

predict(ridge.mod , s = 50, type = "coefficients")

ridge.mod <- glmnet(x_train, y_train, alpha = 0, lambda = grid , thresh = 1e-12)

ridge.pred <- predict(ridge.mod, s = 4, newx = data.matrix(x_test))

# MSE of ridge regression with λ=50

mean((ridge.pred - y_test)^2)

# MSE of ridge regression with λ=50 is greater than MSE of multiple linear regression

# If we had instead simply fit a model with just an intercept, the MSE will be greater

mean((mean(y_train) - y_test)^2)

ridge.pred <- predict(ridge.mod, s = 1e10, newx = data.matrix(x_test))

mean((ridge.pred - y_test)^2)

ridge.pred <- predict(ridge.mod, s = 0, newx = data.matrix(x_test), exact = T, x = x_train, y = y_train)

mean((ridge.pred - y_test)^2)

lm(y_train ~ ., x_train)

predict(ridge.mod , s = 0, exact = T, type = "coefficients", x = x_train, y = y_train)

# In general, instead of arbitrarily choosing λ = 4, it would be better to use cross-validation to choose the tuning parameter λ.

set.seed(1)
cv.out <- cv.glmnet(data.matrix(x_train), y_train, alpha = 0)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam

# Best lambda is 0.07591967

ridge.pred <- predict(ridge.mod , s = bestlam , newx = data.matrix(x_test))
mean((ridge.pred - y_test)^2)

# MSE of ridge regression with is greater than MSE of multiple linear regression

out <- glmnet(x, y, alpha = 0)
predict(out , type = "coefficients", s = bestlam)

# Lasso (least absolute shrinkage and selection operator)

lasso.mod <- glmnet(x_train, y_train, alpha = 1, lambda = grid)
plot(lasso.mod)

set.seed (1)
cv.out <- cv.glmnet(data.matrix(x_train), y_train, alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod , s = bestlam, newx = data.matrix(x_test))
mean((lasso.pred - y_test)^2)

# Lasso MSE is less than Ridge regression MSE, but greater than MLR (OLS fitting)

out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out , type = "coefficients", s = bestlam)
lasso.coef


# The lasso has a substantial advantage over ridge regression in that the resulting coefficient estimates are sparse.







