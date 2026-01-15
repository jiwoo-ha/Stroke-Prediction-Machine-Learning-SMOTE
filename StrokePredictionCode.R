library(themis)
library(recipes)
library(caret)
library(pROC)
library(e1071)
library(MLmetrics)
library(randomForest)
library(xgboost)
library(lightgbm)
library(catboost)

df <- read.csv("dataset.csv", stringsAsFactors = FALSE)

# data preprocessing
df <- df[!is.na(df$gender), ] 
df$smoking_status[is.na(df$smoking_status) & df$age >= 18] <- "Unknown_adult"
df$smoking_status[is.na(df$smoking_status) & df$age < 18]  <- "Unknown_child"

df$age_group <- cut(df$age, breaks = c(-Inf, 13, 18, Inf),
                    labels = c("child", "teen", "adult"), right = FALSE)

bmi_medians <- df %>%
  filter(!is.na(bmi)) %>%
  group_by(age_group) %>%
  summarise(median_bmi = median(bmi), .groups = 'drop')

for (g in levels(df$age_group)) {
  val <- bmi_medians$median_bmi[bmi_medians$age_group == g]
  df$bmi[is.na(df$bmi) & df$age_group == g] <- val
}

df$age_group <- NULL
df$stroke <- as.factor(df$stroke)

# Train/Test split
set.seed(123)
train_index <- createDataPartition(df$stroke, p = 0.8, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

# Change categorical variables to factors
cat_vars <- c("gender", "hypertension", "heart_disease", "ever_married",
              "work_type", "Residence_type", "smoking_status")

train[cat_vars] <- lapply(train[cat_vars], factor)

# Apply SMOTE-NC
rec <- recipe(stroke ~ ., data = train) %>%
  step_rm(id) %>%
  step_smotenc(stroke, over_ratio = 1, neighbors = 5)

rec_prep <- prep(rec, training = train)
train_balanced <- bake(rec_prep, new_data = NULL)

# add terms
add_features <- function(data) {
  
  # change categorical variables to numeric
  data$hypertension_num  <- as.numeric(as.character(data$hypertension))
  data$heart_disease_num <- as.numeric(as.character(data$heart_disease))
  
  # new terms
  data$ht_glucose  <- data$hypertension_num * data$avg_glucose_level
  data$age_ht      <- data$age * data$hypertension_num
  data$bmi_age     <- data$bmi * data$age
  data$bmi_glucose <- data$bmi * data$avg_glucose_level
  data$is_smoker   <- ifelse(data$smoking_status == "smokes", 1, 0)
  data$age_smokes  <- data$age * data$is_smoker
  data$bmi_square <- data$bmi^2
  data$hypertension <- factor(data$hypertension_num, levels = c(0,1))
  data$heart_disease <- factor(data$heart_disease_num, levels = c(0,1))
  
  data$hypertension_num <- NULL
  data$heart_disease_num <- NULL
  
  return(data)
}

train          <- add_features(train)
train_balanced <- add_features(train_balanced)
test           <- add_features(test)

# change categorical variables to factor again
train$is_smoker <- factor(train$is_smoker)
train_balanced$is_smoker <- factor(train_balanced$is_smoker)
test$is_smoker <- factor(test$is_smoker)

# make stroke as factor
train$stroke <- factor(train$stroke)
train_balanced$stroke <- factor(train_balanced$stroke)
test$stroke <- factor(test$stroke)

# scaling
scale_vars <- c("age","bmi","avg_glucose_level", "bmi_square",
                "ht_glucose","age_ht","bmi_age",
                "bmi_glucose","age_smokes")

scaler <- preProcess(train[, scale_vars], method = c("center", "scale"))

train_balanced[, scale_vars] <- predict(scaler, train_balanced[, scale_vars])
test[, scale_vars]           <- predict(scaler, test[, scale_vars])
train[, scale_vars]          <- predict(scaler, train[, scale_vars])  

# full formula for logistic regression
formula_logit <- stroke ~ 
  gender + hypertension + heart_disease + ever_married +
  work_type + Residence_type + smoking_status + is_smoker +
  age + bmi + avg_glucose_level + bmi_square +
  ht_glucose + age_ht + bmi_age + bmi_glucose + age_smokes

# logistic regression full model
full_model <- glm(formula_logit, data = train, family = binomial)
summary(full_model)

# Stepwise Selection (BIC)
n <- nrow(train)
step_model_bic <- stats::step(full_model, direction = "both", k = log(n), trace = TRUE)
summary(step_model_bic)

# final model using original data
final_model1 <- glm(stroke ~ age + hypertension + heart_disease + 
                      is_smoker + bmi + bmi_square + bmi_age, 
                    data = train, family = binomial)
summary(final_model1)

# final model using balanced data
final_model2 <- glm(stroke ~ age + hypertension + heart_disease + 
                      is_smoker + bmi + bmi_square + bmi_age, 
                    data = train_balanced, family = binomial)
summary(final_model2)

# prediction
logit_probs <- predict(final_model2, newdata = test, type = "response")
logit_pred <- ifelse(logit_probs >= 0.5, 1, 0)
logit_pred <- factor(logit_pred, levels = c(0, 1))
actual <- test$stroke

# eval
cat("\n=== Logistic Regression Performance ===\n")
lr_cm <- confusionMatrix(logit_pred, actual, positive = "1")
print(lr_cm)
cat("AUC:", auc(roc(actual, logit_probs)), "\n")
cat("F1 Score:", F1_Score(y_pred = logit_pred, y_true = actual, positive = "1"), "\n")

# data preprocessing for machine learning models
test$id <- NULL

y_train <- as.numeric(train_balanced$stroke) - 1
y_test  <- as.numeric(test$stroke) - 1

remove_cols <- c("stroke", "ht_glucose","age_ht","bmi_age",
                 "bmi_glucose","age_smokes", "is_smoker")
X_train <- train_balanced[, !names(train_balanced) %in% remove_cols]
X_test  <- test[, !names(test) %in% remove_cols]


X_train_mat <- data.matrix(X_train)
X_test_mat  <- data.matrix(X_test)

# Random Forest
set.seed(42)
rf_model <- randomForest(
  x = X_train_mat, 
  y = as.factor(y_train),
  ntree = 300,
  mtry = sqrt(ncol(X_train_mat)),
  importance = TRUE
)

# prediction
rf_probs <- predict(rf_model, X_test_mat, type = "prob")[, 2]
rf_pred <- ifelse(rf_probs >= 0.5, 1, 0)
rf_pred <- factor(rf_pred, levels = c(0, 1))
y_test_factor <- factor(y_test, levels = c(0, 1))

# eval
rf_cm <- confusionMatrix(rf_pred, y_test_factor, positive = "1")
print(rf_cm)
rf_roc <- roc(y_test, rf_probs)
cat("AUC:", auc(rf_roc), "\n")
cat("F1 Score:", F1_Score(rf_pred, y_test_factor, positive = "1"), "\n")

# Random Forest Feature Importance
rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(
  Feature = rownames(rf_importance),
  Importance = rf_importance[, "MeanDecreaseGini"]
)
rf_importance_df <- rf_importance_df[order(-rf_importance_df$Importance), ]
print(rf_importance_df)

# CatBoost
for (col in cat_vars) {
  train[[col]]          <- factor(train[[col]])
  train_balanced[[col]] <- factor(train_balanced[[col]], levels = levels(train[[col]]))
  test[[col]]           <- factor(test[[col]], levels = levels(train[[col]]))
  
  X_train[[col]] <- factor(X_train[[col]], levels = levels(train[[col]]))
  X_test[[col]]  <- factor(X_test[[col]], levels = levels(train[[col]]))
}

cat_features1 <- which(sapply(X_train, is.factor))

train_pool <- catboost.load_pool(
  data = X_train, label = y_train, cat_features = cat_features1
)

test_pool <- catboost.load_pool(
  data = X_test, label = y_test, cat_features = cat_features1
)

cat_params <- list(
  loss_function = "Logloss",
  eval_metric = "AUC",
  iterations = 200,
  learning_rate = 0.1,
  depth = 6,
  l2_leaf_reg = 3,
  border_count = 128,
  random_seed = 42,
  logging_level = "Silent",
  early_stopping_rounds = 20
)

set.seed(42)
cat_model <- catboost.train(
  train_pool,
  test_pool,
  params = cat_params
)

# prediction
cat_probs <- catboost.predict(cat_model, test_pool, prediction_type = "Probability")
cat_pred <- ifelse(cat_probs >= 0.5, 1, 0)
cat_pred <- factor(cat_pred, levels = c(0, 1))

# eval
cat("\n=== CatBoost Performance ===\n")
cat_cm <- confusionMatrix(cat_pred, y_test_factor, positive = "1")
print(cat_cm)
cat_roc <- roc(y_test, cat_probs)
cat("AUC:", auc(cat_roc), "\n")
cat("F1 Score:", F1_Score(cat_pred, y_test_factor, positive = "1"), "\n")

# CatBoost Feature Importance
cat_importance <- catboost.get_feature_importance(cat_model, 
                          pool = test_pool, type = "FeatureImportance")
cat_importance_df <- data.frame(
  Feature = colnames(X_train),
  Importance = cat_importance
)
cat_importance_df <- cat_importance_df[order(-cat_importance_df$Importance), ]
print(cat_importance_df)

# XGBoost
X_train_xgb <- X_train
X_test_xgb  <- X_test

# Gender: Male=0, Female=1
X_train_xgb$gender <- as.numeric(factor(X_train$gender, levels = c("Male", "Female"))) - 1
X_test_xgb$gender  <- as.numeric(factor(X_test$gender, levels = c("Male", "Female"))) - 1

# Ever Married: No=0, Yes=1
X_train_xgb$ever_married <- as.numeric(factor(X_train$ever_married, levels = c("No", "Yes"))) - 1
X_test_xgb$ever_married  <- as.numeric(factor(X_test$ever_married, levels = c("No", "Yes"))) - 1

# Work Type: children=0, Never_worked=1, Govt_job=2, Private=3, Self-employed=4
work_levels <- c("children", "Never_worked", "Govt_job", "Private", "Self-employed")
X_train_xgb$work_type <- as.numeric(factor(X_train$work_type, levels = work_levels)) - 1
X_test_xgb$work_type  <- as.numeric(factor(X_test$work_type, levels = work_levels)) - 1

# Residence Type: Rural=0, Urban=1
X_train_xgb$Residence_type <- as.numeric(factor(X_train$Residence_type, levels = c("Rural", "Urban"))) - 1
X_test_xgb$Residence_type  <- as.numeric(factor(X_test$Residence_type, levels = c("Rural", "Urban"))) - 1

# Smoking Status: never smoked=0, Unknown_child=1, formerly smoked=2, Unknown_adult=3, smokes=4
smoke_levels <- c("never smoked", "Unknown_child", "formerly smoked", "Unknown_adult", "smokes")
X_train_xgb$smoking_status <- as.numeric(factor(X_train$smoking_status, levels = smoke_levels)) - 1
X_test_xgb$smoking_status  <- as.numeric(factor(X_test$smoking_status, levels = smoke_levels)) - 1

X_train_mat <- data.matrix(X_train_xgb)
X_test_mat  <- data.matrix(X_test_xgb)
dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train)
dtest <- xgb.DMatrix(data = X_test_mat, label = y_test)
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  gamma = 0
)

set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest),
  print_every_n = 50,
  early_stopping_rounds = 20,
  verbose = 1
)

# prediction
xgb_probs <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_probs >= 0.5, 1, 0)
xgb_pred <- factor(xgb_pred, levels = c(0, 1))

# eval
cat("\n=== XGBoost Performance ===\n")
xgb_cm <- confusionMatrix(xgb_pred, y_test_factor, positive = "1")
print(xgb_cm)
xgb_roc <- roc(y_test, xgb_probs)
cat("AUC:", auc(xgb_roc), "\n")
cat("F1 Score:", F1_Score(xgb_pred, y_test_factor, positive = "1"), "\n")
# XGBoost Feature Importance
xgb_importance <- xgb.importance(model = xgb_model)
print(xgb_importance)

# LightGBM
lgb_train <- lgb.Dataset(
  data = X_train_mat,
  label = y_train
)

lgb_valid <- lgb.Dataset(
  data = X_test_mat,
  label = y_test
)

lgb_params <- list(
  objective = "binary",
  metric = "auc",
  num_leaves = 31,
  learning_rate = 0.1,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  min_data_in_leaf = 20,
  lambda_l1 = 0.1,
  lambda_l2 = 0.1,
  verbose = -1
)

set.seed(42)
lgb_model <- lgb.train(
  params = lgb_params,
  data = lgb_train,
  nrounds = 200,
  valids = list(test = lgb_valid),
  early_stopping_rounds = 20,
  verbose = 1
)

# prediction
lgb_probs <- predict(lgb_model, X_test_mat)
lgb_pred <- ifelse(lgb_probs >= 0.5, 1, 0)
lgb_pred <- factor(lgb_pred, levels = c(0, 1))

# eval
cat("\n=== LightGBM Performance ===\n")
lgb_cm <- confusionMatrix(lgb_pred, y_test_factor, positive = "1")
print(lgb_cm)
lgb_roc <- roc(y_test, lgb_probs)
cat("AUC:", auc(lgb_roc), "\n")
cat("F1 Score:", F1_Score(lgb_pred, y_test_factor, positive = "1"), "\n")
# LightGBM Feature Importance
lgb_importance <- lgb.importance(lgb_model)
print(lgb_importance)


# Final Results
results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"),
  AUC = c(
    auc(roc(actual, logit_probs)),
    auc(rf_roc),
    auc(xgb_roc),
    auc(lgb_roc),
    auc(cat_roc)
  ),
  F1_Score = c(
    F1_Score(logit_pred, actual, positive = "1"),
    F1_Score(rf_pred, y_test_factor, positive = "1"),
    F1_Score(xgb_pred, y_test_factor, positive = "1"),
    F1_Score(lgb_pred, y_test_factor, positive = "1"),
    F1_Score(cat_pred, y_test_factor, positive = "1")
  ),
  Sensitivity = c(
    as.numeric(lr_cm$byClass["Sensitivity"]),
    rf_cm$byClass["Sensitivity"],
    xgb_cm$byClass["Sensitivity"],
    lgb_cm$byClass["Sensitivity"],
    cat_cm$byClass["Sensitivity"]
  ),
  Specificity = c(
    as.numeric(confusionMatrix(logit_pred, actual, positive = "1")$byClass["Specificity"]),
    rf_cm$byClass["Specificity"],
    xgb_cm$byClass["Specificity"],
    lgb_cm$byClass["Specificity"],
    cat_cm$byClass["Specificity"]
  )
)

results$AUC <- round(results$AUC, 4)
results$F1_Score <- round(results$F1_Score, 4)
results$Sensitivity <- round(results$Sensitivity, 4)
results$Specificity <- round(results$Specificity, 4)

print(results)