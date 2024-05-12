

# 1. Import and Explore the data

#Import all the necessary libraries**


# Install and load the dplyr package for - data manipulation
install.packages('dplyr')
library(dplyr)

# Install and load the caret package for - machine learning tasks
install.packages('caret')
library(caret)

# Install and load the corrplot package for - plot the correlation
install.packages("corrplot")
library(corrplot)

# Install and load the caTools package for - data manipulation and analysis. 
install.packages('caTools')
library(caTools)

# Install and load the glmnet package for - lasso and ridge regression models
install.packages('glmnet')
library(glmnet)

# Install and load the glmnet package for - xgboost model
install.packages('xgboost')
library(xgboost)

# Install and load the tibble package for - 
install.packages('tibble')
library(tibble)

# Install and load the tibble package for - 
install.packages('tidyr')
library(tidyr)

# Install and load the tibble package for - 
install.packages('ggcorrplot')
library(ggcorrplot)

###Read the dataset

## lets read the training data first,
# Define the path of the file
file_path <- "/content/sample_data/housing_train.csv"

# Read the file using read.csv()
train_data <- read.csv(file_path, header = TRUE, stringsAsFactors = FALSE)

dim(train_data)

head(train_data)

## 2. Exploratory Data Analysis (EDA)

# Summary statistics
summary(train_data)

## lets check our predictive variable i.e SalePrice
# Create a histogram of the SalePrice variable
hist(train_data$SalePrice, main = "Histogram of SalePrice", xlab = "SalePrice")

# Boxplot of SalePrice variable
boxplot(train_data$SalePrice)

# Scatterplot of SalePrice vs. GrLivArea variables
plot(train_data$GrLivArea, train_data$SalePrice)

## Lets plot scatterplot to check outliers -

plot(train_data$OverallQual, train_data$SalePrice)

# 3. Clean and Preprocess the dataset

# Check for missing values -train
nulls_train = colSums(is.na(train_data))
print(nulls_train)

#removing columns with NA values more than 10% of the data
set.seed(123)
colMeans(is.na(train_data)) > .10
train_data <- train_data[, colMeans(is.na(train_data)) <= .10]
dim(train_data)

#replace NA values of numerical column with mean of column and categorical column with mode of column
for (col in names(train_data)) {
  if (is.numeric(train_data[[col]])) {
    train_data[[col]] <- ifelse(is.na(train_data[[col]]), mean(train_data[[col]], na.rm = TRUE), train_data[[col]])
  } else {
    train_data[[col]] <- ifelse(is.na(train_data[[col]]), mode(train_data[[col]]), train_data[[col]])
  }
}

# Check for null values column-wise
new_nulls = colSums(is.na(train_data))
print(new_nulls)
dim(train_data)
train_data <- as.data.frame(train_data)

# Check for duplicates
duplicated_rows <- duplicated(train_data)

# Count the number of duplicates
sum(duplicated_rows)

dim(train_data)

#Outlier Removal

# Select the numerical columns in the dataset
num_cols <- sapply(train_data, is.numeric)
num_df <- train_data[, num_cols]
dim(num_df)

# Define a threshold for outlier detection (e.g., 3 standard deviations from the mean)
outlier_threshold <- 4.5

# Identify outlier rows in numerical columns

z_scores <- apply(num_df, 2, function(x) abs(scale(x, center = TRUE, scale = TRUE)))
outlier_rows_num <- apply(z_scores, 1, function(x) any(x > outlier_threshold))

# Identify outlier rows in categorical columns
cat_cols <- sapply(train_data, is.factor)
cat_df <- train_data[, cat_cols]
outlier_rows_cat <- apply(cat_df, 1, function(x) any(x == ""))

# Combine outlier rows from both numerical and categorical columns
outlier_rows <- outlier_rows_num | outlier_rows_cat

# Print the number of detected outliers for each column type
cat("Number of outliers detected:\n")
cat(sprintf("Total: %d\n", sum(outlier_rows)))

# Remove the outlier rows from the dataset
train_data_clean <- train_data[!outlier_rows,]

dim(train_data_clean)

# 4. Feature Engineering

train_data <- data.matrix(train_data_clean)
dim(train_data)
head(train_data)

cor_matrix <- cor(train_data)
cor_matrix

## find the correlation between the variables and drop based on condition
d2 <- train_data %>% 
  as.matrix %>%
  cor %>%
  as.data.frame %>%
  rownames_to_column(var = 'var1') %>%
  gather(var2, value, -var1)
d2

to_remove <- filter(d2, ((value > 0.8 | value < -0.8) & (var1 != var2)))
print(to_remove)

# based on high correlation I have choosen following five columns to drop
#train_data <- train_data[ , !(colnames(train_data) %in% c('Exterior1st','GarageArea','Exterior2nd','TotRmsAbvGrd','X1stFlrSF'))]
#dim(train_data)

# based on high correlation I have choosen following eight columns to drop
train_data <- train_data[ , !(colnames(train_data) %in% c('Exterior1st','GarageArea','GrLivArea','Exterior2nd','TotRmsAbvGrd','GarageCars','GarageCond','GarageQual'))]
dim(train_data)

# lets see  the correlation matrix again
corr_matrix <- cor(train_data)

# Print the correlation matrix
print(corr_matrix)


#5. Data Preparation for model

# Remove the ID column
train_data <- train_data[, !(colnames(train_data) == "Id")]

head(train_data)
dim(train_data)

str(train_data)

# 6. Model Development

#Split the dataset into train and validation


train_data <- data.frame(train_data)

# Split the data into training and validation sets
set.seed(123)
sample <- sample(c(TRUE, FALSE), nrow(train_data), 
                 replace=TRUE, prob=c(0.7,0.3))

train  <- train_data[sample, ]
validation   <- train_data[!sample, ]

dim(train)
dim(validation)

head(train)
head(validation)

# 7. Lets Fit Predict and generate resuts on different models

# i. Linear Regression

##Fit the Model on train data


fit.lm <- lm(formula = SalePrice ~ ., data = train)

###Prediction on validation

pred.lm <- predict(fit.lm, newdata = validation)

actuals_preds.lm <- data.frame(cbind(actuals=validation$SalePrice, predicteds=pred.lm))
head(actuals_preds.lm)

### Performance metrics

# Calculate the RMSE
rmse_lm <- sqrt(mean((pred.lm - validation$SalePrice)^2))
print(rmse_lm)

# Accuracy 
accuracy_lm <- cor(actuals_preds.lm, validation$SalePrice)^2
print(paste("Accuracy on validation set:", round(accuracy_lm * 100, 2), "%"))

# plot the graph 
plot(pred.lm, validation$SalePrice, xlab = "Predicted SalePrice", ylab = "Actual SalePrice",
     main = "Linear Regression Predictions vs. Validation Data")
abline(0, 1, col = "red")

# ii. Elastic Net

##Fit the Model on train data

set.seed(42)
cv_5 = trainControl(method = "cv", number = 5)

fit.elnet = train(SalePrice ~ ., data = train, method = "glmnet", trControl = cv_5)
summary(fit.elnet)

## Prediction on validation** """

pred.elnet <- predict(fit.elnet, newdata = validation)
#pred.elnet

actuals_preds.elnet <- data.frame(cbind(actuals=validation$SalePrice, predicteds = pred.elnet))
head(actuals_preds.elnet)

## Performance metrics

# Calculate the RMSE
rmse_elnet <- sqrt(mean((pred.elnet - validation$SalePrice)^2))
print(rmse_elnet)

# Accuracy 
accuracy_elnet <- cor(actuals_preds.elnet, validation$SalePrice)^2
print(paste("Accuracy on validation set:", round(accuracy_elnet * 100, 2), "%"))

# plot the graph 
plot(pred.elnet, validation$SalePrice, xlab = "Predicted SalePrice", ylab = "Actual SalePrice",
     main = "Elastic Model Predictions vs. Validation Data")
abline(0, 1, col = "red")

## iii. Lasso Regression

## Fit the Model on train data

# Fit the lasso regression model
fit.lasso <- glmnet(x = model.matrix(SalePrice ~ ., data = train), 
                    y = train$SalePrice, 
                    alpha = 1, # Lasso regression
                    lambda = 0.1) # Value of lambda

## Prediction on validation

# Predict on the test set
pred.lasso <- predict(fit.lasso, newx = model.matrix(SalePrice ~ ., data = validation))

lasso_actuals_preds <- data.frame(cbind(actuals=validation$SalePrice, predicteds = pred.lasso))
head(lasso_actuals_preds)

## Performance metrics


# Calculate the RMSE
rmse_lasso <- sqrt(mean((pred.lasso - validation$SalePrice)^2))
print(rmse_lasso)

# Accuracy 
accuracy_lasso <- cor(lasso_actuals_preds, validation$SalePrice)^2
print(paste("Accuracy on validation set:", round(accuracy_lasso * 100, 2), "%"))

# plot the graph 
plot(pred.lasso, validation$SalePrice, xlab = "Predicted SalePrice", ylab = "Actual SalePrice",
     main = "Lasso Model Predictions vs. Validation Data")
abline(0, 1, col = "red")

# iv. Ridge Regression

##Fit the Model on train data


# Fit the ridge regression model
fit.ridge <- glmnet(x = model.matrix(SalePrice ~ ., data = train), 
                    y = train$SalePrice, 
                    alpha = 0, # Ridge regression
                    lambda = 0.1) # Value of lambda

## Prediction on validation

# Predict on the test set
pred.ridge <- predict(fit.ridge, newx = model.matrix(SalePrice ~ ., data = validation))

ridge_actuals_preds <- data.frame(cbind(actuals=validation$SalePrice, predicteds = pred.ridge))
head(ridge_actuals_preds)

## Performance metrics
# Calculate the RMSE
rmse_ridge <- sqrt(mean((pred.ridge - validation$SalePrice)^2))
print(rmse_ridge)

# Accuracy 
accuracy_ridge <- cor(ridge_actuals_preds, validation$SalePrice)^2
print(paste("Accuracy on validation set:", round(accuracy_ridge * 100, 2), "%"))

# plot the graph 

plot(pred.ridge, validation$SalePrice, xlab = "Predicted SalePrice", ylab = "Actual SalePrice",
     main = "Ridge Model Predictions vs. Validation Data")
abline(0, 1, col = "red")

# v. Decision Tree

##Fit the Model on train data


library(rpart)
fit.dectree <- rpart(SalePrice ~ ., data = train)
plot(fit.dectree)
text(fit.dectree, use.n = TRUE)

## Prediction on validation**

#pred.dectree <- predict(fit.dectree, newx = model.matrix(SalePrice ~ ., data = validation))
pred.dectree <- predict(fit.dectree, newdata = validation)

dectree_actuals_preds <- data.frame(cbind(actuals=validation$SalePrice, predicteds = pred.dectree))
head(dectree_actuals_preds)

## Performance metrics

# Calculate the RMSE
rmse_dectree <- sqrt(mean((pred.dectree - validation$SalePrice)^2))
print(rmse_dectree)

# Accuracy 
accuracy_dectree <- cor(pred.dectree, validation$SalePrice)^2
print(paste("Accuracy on test set:", round(accuracy_dectree * 100, 2), "%"))

plot(pred.dectree, validation$SalePrice, xlab = "Predicted SalePrice", ylab = "Actual SalePrice",
     main = "Decision Tree Model Predictions vs. Validation Data")
abline(0, 1, col = "red")

# vi. Random Forest

## Fit the Model on train data


install.packages("randomForest")
library(randomForest)

fit.rf <- randomForest(SalePrice ~ ., data = train)
print(fit.rf)

### Prediction on validation

pred.rf <- predict(fit.rf, newdata = validation)

rf_actuals_preds <- data.frame(cbind(actuals=validation$SalePrice, predicteds = pred.rf))
head(rf_actuals_preds)

## Performance metrics

# Calculate the RMSE
rmse_rf <- sqrt(mean((pred.rf - validation$SalePrice)^2))
print(rmse_rf)

# Accuracy 
accuracy_rf <- cor(pred.rf, validation$SalePrice)^2
print(paste("Accuracy on test set:", round(accuracy_rf * 100, 2), "%"))

plot(pred.rf, validation$SalePrice, xlab = "Predicted SalePrice", ylab = "Actual SalePrice",
     main = "Decision Tree Model Predictions vs. Validation Data")
abline(0, 1, col = "red")

# vii. Gradient Boost

##Fit the Model on train data**


install.packages("gbm")
library(gbm)

# Define hyperparameters to tune
gbm_grid <- expand.grid(
  n.trees = seq(50, 200, 50),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.01, 0.1, 0.2),
  n.minobsinnode = c(5, 10, 20)
)

# Train model using grid search
gbm_model <- train(
  x = as.matrix(train[, -ncol(train)]),
  y = train$SalePrice,
  method = "gbm",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = gbm_grid,
  metric = "RMSE"
)

## Prediction on validation

# Make predictions on the test set
pred.gbm<- predict(gbm_model, newdata = validation)

gbm_actuals_preds <- data.frame(cbind(actuals=validation$SalePrice, predicteds = pred.gbm))
head(gbm_actuals_preds)

## Performance metrics

# Calculate the RMSE
rmse_gbm <- sqrt(mean((pred.gbm- validation$SalePrice)^2))
print(rmse_gbm)

# Accuracy 
accuracy_gbm <- cor(pred.gbm, validation$SalePrice)^2
print(paste("Accuracy on test set:", round(accuracy_gbm * 100, 2), "%"))

plot(pred.gbm, validation$SalePrice, xlab = "Predicted SalePrice", ylab = "Actual SalePrice",
     main = "gbm Model Predictions vs. Validation Data")
abline(0, 1, col = "red")

## viii. Xgboost

##Fit the Model on train data


# Define the tuning parameter grid
tune_grid <- expand.grid(
  nrounds = 100,
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1),
  gamma = 0,
  colsample_bytree = seq(0.5, 0.9, 0.1),
  min_child_weight = c(1, 3, 5),
  subsample = seq(0.5, 0.9, 0.1)
)

# Train the XGBoost model using caret
xgb_model <- train(
  x = as.matrix(train[, -ncol(train)]),
  y = train$SalePrice,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = tune_grid,
  metric = "RMSE"
)

## Prediction on validation

# Make predictions on the test set
pred.xgb <- predict(xgb_model, newdata = validation)

xgb_actuals_preds <- data.frame(cbind(actuals=validation$SalePrice, predicteds = pred.xgb))
head(xgb_actuals_preds)

## Performance metrics

# Calculate the RMSE
rmse_xgb <- sqrt(mean((pred.xgb - validation$SalePrice)^2))
print(rmse_xgb)

# Accuracy 
accuracy_xgb <- cor(pred.xgb, validation$SalePrice)^2
print(paste("Accuracy on test set:", round(accuracy_xgb * 100, 2), "%"))

plot(pred.xgb, validation$SalePrice, xlab = "Predicted SalePrice", ylab = "Actual SalePrice",
     main = "XGB Model Predictions vs. Validation Data")
abline(0, 1, col = "red")

# 8. Model Analysis and best model selection

# Create a data frame with the model names and RMSE values
rmse_df <- data.frame(Model = c("fit.lm", "fit.elnet", "fit.lasso", "fit.ridge", "pred.dectree", "fit.rf","pred.gbm", "pred.xgb"),
                      RMSE = c(rmse_lm, rmse_elnet, rmse_lasso, rmse_ridge, rmse_dectree, rmse_rf, rmse_gbm, rmse_xgb ))

# Print the data frame
print(rmse_df)

# Create a data frame with the model names and accuracy values
accuracy_df <- data.frame(Model = c("fit.lm", "fit.elnet", "fit.lasso", "fit.ridge", "pred.dectree", "fit.rf", "pred.gbm", "pred.xgb"),
                          Accuracy = c(accuracy_lm[2,1], accuracy_elnet[2,1], accuracy_lasso[2,1], accuracy_ridge[2,1],
                                       accuracy_dectree, accuracy_rf, accuracy_gbm, accuracy_xgb))

# Print the data frame
print(accuracy_df)

# 9. Sale prices of housing on test dataset

## Lets perform same preprocessing steps on test data too**


## now lets read the test data
test_data = read.csv('/content/sample_data/housing_test.csv')
dim(test_data)

head(test_data)

#replace NA values of numerical column with mean of column and categorical column with mode of column
for (col in names(test_data)) {
  if (is.numeric(test_data[[col]])) {
    test_data[[col]] <- ifelse(is.na(test_data[[col]]), mean(test_data[[col]], na.rm = TRUE), test_data[[col]])
  } else {
    test_data[[col]] <- ifelse(is.na(test_data[[col]]), mode(test_data[[col]]), test_data[[col]])
  }
}

# Check for null values column-wise
new_nulls = colSums(is.na(test_data))
print(new_nulls)
test_data <- as.data.frame(test_data)
dim(test_data)

# Check for duplicates
duplicated_rows <- duplicated(test_data)

# Count the number of duplicates
sum(duplicated_rows)

selected_cols <- c('Id', 'MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition')
test_data <- test_data[selected_cols]
dim(test_data)

# Select the numerical columns in the dataset
num_cols <- sapply(test_data, is.numeric)
num_df <- test_data[, num_cols]
dim(num_df)

# Define a threshold for outlier detection (e.g., 3 standard deviations from the mean)
outlier_threshold <- 4.5

# Identify outlier rows in numerical columns

z_scores <- apply(num_df, 2, function(x) abs(scale(x, center = TRUE, scale = TRUE)))
outlier_rows_num <- apply(z_scores, 1, function(x) any(x > outlier_threshold))

# Identify outlier rows in categorical columns
cat_cols <- sapply(test_data, is.factor)
cat_df <- test_data[, cat_cols]
outlier_rows_cat <- apply(cat_df, 1, function(x) any(x == ""))

# Combine outlier rows from both numerical and categorical columns
outlier_rows <- outlier_rows_num | outlier_rows_cat

# Print the number of detected outliers for each column type
cat("Number of outliers detected:\n")
cat(sprintf("Total: %d\n", sum(outlier_rows)))

# Remove the outlier rows from the dataset
test_data_clean <- test_data[!outlier_rows,]

dim(test_data_clean)

# remove same correlated variables as thst from train data
test_data <- test_data_clean[ , !(colnames(test_data_clean) %in% c('Exterior1st','GarageArea','GrLivArea','Exterior2nd','TotRmsAbvGrd','GarageCars','GarageCond','GarageQual'))]
dim(test_data)

# Remove the ID column
test_data_new <- test_data[, !(colnames(test_data) == "Id")]
head(test_data_new)

test_data <- data.frame(test_data_new)
dim(test_data)

## choose the final model and insert below -

# Predict using the gbm model:
test_prediction_final_model <- predict(gbm_model, newdata = test_data)
#test_prediction_final_model

# create a csv and save the results
write.csv(data.frame(test_prediction_final_model), "housing_test_results.csv", row.names = FALSE)

# Read the CSV file into a data frame
housing_test_results <- read.csv("housing_test_results.csv")

# View the first few rows of the data frame
head(housing_test_results)