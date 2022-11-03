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
ggcorrplot(cor, lab = TRUE)

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











