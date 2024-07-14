# Load the dataset
mlr <- read.csv(choose.files())

# Get the required packages
library(psych)
library(DataExplorer)
library(car) # scatterplot, vif
library(lmtest) # Autocorrelation 
library(MASS) # Step-wise regression
library(Metrics) # Loss/Cost function

# Perform EDA
# Understand data structure
str(mlr)

# Missingness analysis
summary(mlr)
is.na(mlr)
plot_missing(mlr)

# Understand distributions and correlations
pairs.panels(mlr)
plot_histogram(mlr)
plot_correlation(mlr)
plot_density(mlr)

# Data partitioning
set.seed(1234)
mlr_mixed <- mlr[order(runif(nrow(mlr))), ]
mlr_training <- mlr_mixed[1:floor(0.7 * nrow(mlr)), ]
mlr_testing <- mlr_mixed[(floor(0.7 * nrow(mlr)) + 1):nrow(mlr), ]

# Build a full model
mlr_lm_full <- lm(mpg ~ ., data=mlr_training)
summary(mlr_lm_full)

# Select best features
mlr_step <- stepAIC(mlr_lm_full, direction="backward")


# Build the Reduced Model
mlr_reduced <- lm(formula(mlr_step), data=mlr_training)
summary(mlr_reduced)



# Model diagnostics
plot(mlr_reduced)

# Check for multi-collinearity
vif(mlr_reduced)


# Auto-correlation of residuals (Durbin-Watson Test)
durbinWatsonTest(mlr_reduced)
dwtest(mlr_reduced)

# Component Residual Plot to check linearity
crPlots(mlr_reduced)

# Non-constant Variance
ncvTest(mlr_reduced)



# Use model for prediction
mlr_prediction <- predict(mlr_reduced, newdata=mlr_testing)

# Actual Vs Predicted
newtest_pred <- cbind(mlr_testing, mlr_prediction)
head(newtest_pred)

# Mean Squared Error
mse(mlr_testing$mpg, mlr_prediction)

# Function to calculate Adjusted R-squared
calculate_adjusted_r_squared <- function(r_squared, n, p) {
  1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
}

# Number of observations
n <- nrow(mlr_training)

# Number of predictors
p <- length(coef(mlr_reduced)) - 1

# Adjusted R-squared for the reduced model
r_squared <- summary(mlr_reduced)$r.squared
adjusted_r_squared <- calculate_adjusted_r_squared(r_squared, n, p)
adjusted_r_squared








library(Metrics) # Loss/Cost function
library(glmnet) # Lasso and Ridge regression

# Prepare the data for Lasso regression
x <- model.matrix(mpg ~ ., mlr_training)[,-1]
y <- mlr_training$mpg

# Fit Lasso regression model
lasso_model <- cv.glmnet(x, y, alpha=1)
plot(lasso_model)

# Get the best lambda
best_lambda <- lasso_model$lambda.min

# Fit the final Lasso model with the best lambda
lasso_final <- glmnet(x, y, alpha=1, lambda=best_lambda)
coef(lasso_final)

# Predict using the Lasso model
x_test <- model.matrix(mpg ~ ., mlr_testing)[,-1]
lasso_predictions <- predict(lasso_final, s=best_lambda, newx=x_test)

# Mean Squared Error
mse_lasso <- mse(mlr_testing$mpg, lasso_predictions)
mse_lasso

# Adjusted R-squared for Lasso
r_squared_lasso <- 1 - (sum((lasso_predictions - mlr_testing$mpg)^2) / sum((mlr_testing$mpg - mean(mlr_testing$mpg))^2))
adjusted_r_squared_lasso <- calculate_adjusted_r_squared(r_squared_lasso, n, p)
adjusted_r_squared_lasso





# Fit Ridge regression model
ridge_model <- cv.glmnet(x, y, alpha=0)
plot(ridge_model)

# Get the best lambda
best_lambda_ridge <- ridge_model$lambda.min

# Fit the final Ridge model with the best lambda
ridge_final <- glmnet(x, y, alpha=0, lambda=best_lambda_ridge)
coef(ridge_final)

# Predict using the Ridge model
ridge_predictions <- predict(ridge_final, s=best_lambda_ridge, newx=x_test)

# Mean Squared Error
mse_ridge <- mse(mlr_testing$mpg, ridge_predictions)
mse_ridge

# Adjusted R-squared for Ridge
r_squared_ridge <- 1 - (sum((ridge_predictions - mlr_testing$mpg)^2) / sum((mlr_testing$mpg - mean(mlr_testing$mpg))^2))
adjusted_r_squared_ridge <- calculate_adjusted_r_squared(r_squared_ridge, n, p)
adjusted_r_squared_ridge

