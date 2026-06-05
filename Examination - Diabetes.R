# Examination - Diabetes.R
# Authors: Therese Holmager, Arnav Tandon, Sebastian Winkelmann
# CDC Diabetes Health Indicators

library(mlr3)
library(mlr3learners)
library(mlr3torch)
library(mlr3viz)
library(fastshap)
library(iml)
library(ggplot2)
library(mlr3tuning)
library(mlr3mbo)
library(mlr3pipelines)
library(paradox)
library(future)
library(tidyverse)
library(data.table)
library(iml)
library(ggplot2)
library(ranger)

future::plan("multisession")

#Import dataset
diabetes <- read.csv("project/diabetes.csv")
#The dataset has 253680 observations and 22 variables

task = as_task_classif(diabetes, target = "Diabetes_binary")
set.seed(2026)

idx <- sample(nrow(diabetes), 5000)
diabetes_small <- diabetes[idx, ]

# Use table to check if all values match the variable description (e.g. binary variables being 0 or 1) and see if there are missing values.
table(diabetes$Diabetes_binary, useNA = "ifany")
prop.table(table(diabetes$Diabetes_binary, useNA = "ifany"))
table(diabetes$HighBP, useNA = "ifany")
table(diabetes$HighChol, useNA = "ifany")
table(diabetes$CholCheck, useNA = "ifany")
table(diabetes$BMI, useNA = "ifany")
table(diabetes$Smoker, useNA = "ifany")
table(diabetes$Stroke, useNA = "ifany")
table(diabetes$HeartDiseaseorAttack, useNA = "ifany")
table(diabetes$PhysActivity, useNA = "ifany")
table(diabetes$Fruits, useNA = "ifany")
table(diabetes$Veggies, useNA = "ifany")
table(diabetes$HvyAlcoholConsump, useNA = "ifany")
table(diabetes$AnyHealthcare, useNA = "ifany")
table(diabetes$NoDocbcCost, useNA = "ifany")
table(diabetes$GenHlth, useNA = "ifany")
table(diabetes$MentHlth, useNA = "ifany")
table(diabetes$PhysHlth, useNA = "ifany")
table(diabetes$DiffWalk, useNA = "ifany")
table(diabetes$Sex, useNA = "ifany")
table(diabetes$Age, useNA = "ifany")
table(diabetes$Education, useNA = "ifany")
table(diabetes$Income, useNA = "ifany")

#All values are within variable range described on https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators 
#No missing values

sum(duplicated(diabetes))
#24206 duplicated values of 253680 (9.5% of all observations)
#We chose not to delete duplicated rows, as we believe that more than one person with identical values for the 22 variables with few categories may be possible.


#### Choice of models

# We start by defining task in the dataset and splitting it into 5 folds (80% training, 20% test)
task = as_task_classif(diabetes, target = "Diabetes_binary", positive = "1")
set.seed(2026)
splits = partition(task, ratio = 0.8)

task_train = task$clone()$filter(splits$train)
task_test = task$clone()$filter(splits$test)

task_train$positive = "1"
task_test$positive = "1"

# Logistic Regression
my_lr_learner = lrn("classif.log_reg",
                    predict_type = "prob")

graph_learner_lr = as_learner(
  po("encode", method = "treatment", id = "binary_enc") %>>%
    po("scale") %>>%
    my_lr_learner
)

graph_learner_lr$train(task_train)
graph_learner_lr$predict(task_test)$score(msr("classif.ce"))
graph_learner_lr$predict(task_test)$score(msr("classif.sensitivity"))
graph_learner_lr$predict(task_test)$score(msr("classif.specificity"))
graph_learner_lr$predict(task_test)$score(msr("classif.auc"))

# Get coefficients
glm_model <- graph_learner_lr$model$classif.log_reg$model
coef(glm_model)



# Naive model / Baseline with no features
task_train$positive = "1"
task_test$positive = "1"
baseline <- lrn("classif.featureless",
                predict_type = "prob")$train(task_train)
baseline$predict(task_test)$score(msr("classif.ce"))
baseline$predict(task_test)$score(msr("classif.sensitivity"))
baseline$predict(task_test)$score(msr("classif.specificity"))
baseline$predict(task_test)$score(msr("classif.auc"))



# Elastic net. Elastic net combines Lasso (L1) and Ridde (L2) regularization in a model

# First we create a learner. 
my_elasticnet_learner = lrn( "classif.glmnet", predict_type = "prob", 
standardize = FALSE, alpha = to_tune(0, 1), lambda = to_tune(p_dbl(log(1e-5), log(1e1), trafo = exp))) 
# Alternatively lambda = to_tune(1e-5, 1e1, logscale=TRUE) ) 

# We create a graph that converts factor variables into dummy variables, standardizes predictors, and afterwards applies elastic net
graph_learner_elastic_net = as_learner( po("encode", method = "treatment", id = "binary_enc") %>>% po("scale") %>>% my_elasticnet_learner )

# 5-fold cross-validation for elastic net model (train on 4 folds, validate on 1 fold, repeat 5 times). 
set.seed(2026)
instance = tune(
  tuner = tnr("random_search", batch_size=20), 
  task = task_train,
  learner = graph_learner_elastic_net,
  resampling = rsmp("cv", folds = 5), 
  measures = msr("classif.ce"),
  terminator = trm("evals", n_evals = 50) 
)

# Keep an untuned copy of the same graph
graph_learner_elastic_net_auto = graph_learner_elastic_net$clone(deep = TRUE)

# Result on test set
graph_learner_elastic_net$param_set$values = instance$result_learner_param_vals
graph_learner_elastic_net$train(task_train)
graph_learner_elastic_net$predict(task_test)$score(msr("classif.ce"))
graph_learner_elastic_net$predict(task_test)$score(msr("classif.sensitivity"))
graph_learner_elastic_net$predict(task_test)$score(msr("classif.specificity"))
graph_learner_elastic_net$predict(task_test)$score(msr("classif.auc"))

# Get coefficients
glmnet_model <- graph_learner_elastic_net$model$classif.glmnet$model
coef(glmnet_model)



# Full CART decision tree
set.seed(123)


task <- as_task_classif(diabetes_small, target = "Diabetes_binary")
task$positive <- "1"

categorical_selector <- selector_type(c("factor", "ordered"))

make_encoded_learner <- function(learner) {
  as_learner(
    po("encode", method = "treatment",
       affect_columns = categorical_selector) %>>%
      learner
  )
}

# Visualize a large CART tree
my_cart_learner = lrn("classif.rpart", cp = 0, predict_type = "prob")

full_tree = make_encoded_learner(my_cart_learner)
full_tree$train(task)

cart_trained <- full_tree$model$classif.rpart$model
plot(cart_trained, compress = TRUE, margin = 0.1)
text(cart_trained, use.n = TRUE, cex = 0.4) # Adjust cex (font size) as needed


# 5-fold cross-validation for Full CART decision tree 
rr_full = resample(
  task,
  full_tree,
  rsmp("cv", folds = 5)
)

rr_full$aggregate(msr("classif.ce"))
rr_full$aggregate(msr("classif.sensitivity"))
rr_full$aggregate(msr("classif.specificity"))
rr_full$aggregate(msr("classif.auc"))

# Cross-validation path for pruning parameter (weakest-link pruning)
my_cart_learner_cv = lrn("classif.rpart",
                         cp = 0,
                         xval = 5,
                         predict_type = "prob")

graph_learner_cart_cv = make_encoded_learner(my_cart_learner_cv)
graph_learner_cart_cv$train(task)

cart_trained_cv <- graph_learner_cart_cv$model$classif.rpart$model

rpart::plotcp(cart_trained_cv)
rpart::printcp(cart_trained_cv)



# Pruned CART decision tree
cp_table = cart_trained_cv$cptable
min_row = which.min(cp_table[, "xerror"])
one_se_threshold = cp_table[min_row, "xerror"] + cp_table[min_row, "xstd"]

# rpart orders the table from simpler to larger trees, so the first row within the one-SE threshold gives the simplest acceptable tree.
selected_row = which(cp_table[, "xerror"] <= one_se_threshold)[1]
selected_cp = cp_table[selected_row, "CP"]
selected_cp

my_cart_learner_pruned = lrn("classif.rpart",
                             cp = selected_cp,
                             predict_type = "prob")

pruned_tree = make_encoded_learner(my_cart_learner_pruned)
pruned_tree$train(task)

cart_trained_pruned <- pruned_tree$model$classif.rpart$model

plot(cart_trained_pruned, compress = TRUE, margin = 0.1)
text(cart_trained_pruned, use.n = TRUE, cex = 0.5)

# 5-fold cross-validation for pruned tree model
rr_pruned = resample(
  task,
  pruned_tree,
  rsmp("cv", folds = 5)
)

rr_pruned$aggregate(msr("classif.ce"))
rr_pruned$aggregate(msr("classif.sensitivity"))
rr_pruned$aggregate(msr("classif.specificity"))
rr_pruned$aggregate(msr("classif.auc"))

# Random Forest
rf_learner <- lrn(
  "classif.ranger",
  predict_type = "prob",
  num.trees = 200,
  num.threads = 4
)

rf_model <- make_encoded_learner(rf_learner)
rf_model$train(task)

X <- diabetes_small[, colnames(diabetes_small) != "Diabetes_binary"]
y <- diabetes_small$Diabetes_binary

predictor_rf <- Predictor$new(
  model = rf_model,
  data = X,
  y = y,
  type = "prob"
)

# 5-fold cross-validation for random forest
rr_rf = resample(
  task,
  rf_model,
  rsmp("cv", folds = 5)
)

rr_rf$aggregate(msr("classif.ce"))
rr_rf$aggregate(msr("classif.sensitivity"))
rr_rf$aggregate(msr("classif.specificity"))
rr_rf$aggregate(msr("classif.auc"))

# XGboost
xgb_learner = lrn(
  "classif.xgboost",
  predict_type = "prob"
)

xgb_model = make_encoded_learner(xgb_learner)
xgb_model$train(task)

# 5-fold cross-validation for XGboost
rr_xgb = resample(
  task,
  rf_model,
  rsmp("cv", folds = 5)
)

rr_xgb$aggregate(msr("classif.ce"))
rr_xgb$aggregate(msr("classif.sensitivity"))
rr_xgb$aggregate(msr("classif.specificity"))
rr_xgb$aggregate(msr("classif.auc"))

#### Model performance: Benchmark comparison via 5-fold CV

future::plan("sequential")

#Neural Network
search_space <- ps(
  neurons = p_int(lower = 16, upper = 128),
  p = p_dbl(lower = 0.1, upper = 0.5),
  batch_size = p_int(lower = 64, upper = 256),
  epochs = p_int(lower = 20, upper = 50),
  n_layers = p_int(lower = 1, upper = 20),
  activation = p_fct(levels = c(nn_relu, nn_leaky_relu, nn_elu, nn_gelu, nn_tanh))
)

learner <- lrn("classif.mlp",
               predict_type = "prob",
               epochs = 30,
               batch_size = 128,
               neurons = 32,
               p = 0.2,
               optimizer = "adam",
)

resampling <- rsmp("cv", folds = 5)
measure <- msr("classif.recall", id = "recall")
tuner <- tnr("random_search")
at <- auto_tuner(
  tuner = tuner,
  learner = learner_mlp,
  resampling = resampling,
  measure = measure,
  search_space = search_space,
  term_evals = 20 
)

at$train(task_train)
at$tuning_result
best_params <- at$tuning_result$learner_param_vals[[1]]

final_learner <- lrn("classif.mlp",
                     predict_type = "prob",
                     epochs = best_params$epochs,
                     batch_size = best_params$batch_size,
                     neurons = best_params$neurons,
                     p = best_params$p,
                     optimizer = "adam",
                     n_layers = best_params$n_layers
)
final_learner$train(task_train)

final_learner$predict(task_test)$score(msr("classif.ce"))
final_learner$predict(task_test)$score(msr("classif.auc"))
final_learner$predict(task_test)$score(msr("classif.sensitivity"))
final_learner$predict(task_test)$score(msr("classif.specificity"))


# Find threshold that maximises accuracy and recall
val_idx <- sample(task_train$nrow, size = floor(0.2 * task_train$nrow))
val_task <- task_train$clone()$filter(val_idx)

val_pred <- final_learner$predict(val_task)
val_prob <- val_pred$prob[, "1"]
val_true <- val_task$truth()

thresholds <- seq(0.1, 0.9, by = 0.01)
accuracy <- sapply(thresholds, function(th) {
  pred_class <- ifelse(val_prob >= th, 1, 0)
  tp <- sum(pred_class == 1 & val_true == 1)
  fn <- sum(pred_class == 0 & val_true == 1)
  fp <- sum(pred_class == 1 & val_true == 0)
  acc <- tp / sqrt((tp + fn) * (tp + fp))
  acc
})

best_threshold <- thresholds[which.max(accuracy)]


test_task <- as_task_classif(task_test, target = "Diabetes_binary", positive = "1")
test_pred <- final_learner$predict(test_task)
test_prob <- test_pred$prob[, "1"]
test_class <- ifelse(test_prob >= best_threshold, 1, 0)
test_true <- test_task$truth()

cm <- table(Predicted = test_class, Actual = test_true)
recall_test <- cm[2,1] / sum(cm[,1])
precision_test <- cm[2,1] / sum(cm[2,])
ce_test <- (cm[1,1] + cm[2,2]) / sum(cm)

cat(sprintf("Test CE-score: %.8f\n", ce_test))
cat(sprintf("Test recall/sensitivity: %.8f\n", recall_test))
cat(sprintf("Test precision: %.8f\n", precision_test))

# Elastic net (tuned inside CV)
elastic_net_at = auto_tuner(
  learner = graph_learner_elastic_net_auto,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  tuner = tnr("random_search", batch_size = 20),
  terminator = trm("evals", n_evals = 50)
)

# Assign learner id
elastic_net_at$id = "elastic_net"
graph_learner_lr$id = "logistic"
baseline$id = "featureless"
full_tree$id = "cart_large"
pruned_tree$id = "cart_pruned"
rf_model$id = "random_forest"
xgb_model$id = "xgboost"

# Benchmark
set.seed(2026)

benchmark_design = benchmark_grid(
  tasks = list(task),
  learners = list(
    baseline,
    graph_learner_lr,
    elastic_net_at,
    full_tree,
    pruned_tree,
    rf_model,
    xgb_model
  ),
  resamplings = list(rsmp("cv", folds = 5))
)

res = benchmark(
  benchmark_design,
  store_models = TRUE
)

benchmark_results = res$aggregate(list(
  msr("classif.ce"),
  msr("classif.sensitivity"),
  msr("classif.specificity"),
  msr("classif.auc")
))

benchmark_results

#### Post-hoc methods. We identify Random Forest as the best-performing model.

# Individual conditional expectation
ice <- FeatureEffect$new(
  predictor_rf,
  feature = "BMI",
  method = "ice",
  grid.size = 10
)

plot(ice)

# Partial dependence plot
pdp <- FeatureEffect$new(
  predictor_rf,
  feature = "BMI",
  method = "pdp",
  grid.size = 10
)

plot(pdp)

# SHAP
x_interest <- X[1, , drop = FALSE]

shap <- Shapley$new(
  predictor_rf,
  x.interest = x_interest
)

plot(shap)
shap$results

# Permutation feature importance
pfi <- FeatureImp$new(
  predictor_rf,
  loss = "ce",
  compare = "difference"
)

plot(pfi)
pfi$results


# Sensitive variables can be identified by already established correlations (e.g. increased risk of diabetes with increasing BMI) 
# and also by modelling the data. Compare prediction of models without sensitive variables. We would expect less accuracy 
# using the same models without the the sensitive variables.
protected_features <- c("AnyHealthcare", "Education", "Income", "NoDocbcCost")
permitted_features <- setdiff(task_train$feature_names, protected_features)
task_restricted <- task_train$clone()
task_restricted$select(permitted_features)
task_restricted$positive <- "1"
test_restricted <- task_test$clone()
test_restricted$select(permitted_features)


lrn_logreg <- lrn("classif.log_reg", predict_type = "prob")

lrn_glmnet <- lrn("classif.glmnet", predict_type = "prob")
search_space_glmnet <- ps(
  alpha = p_dbl(lower = 0, upper = 1),     # mixing between ridge (0) and lasso (1)
  s = p_dbl(lower = 0.001, upper = 1, logscale = TRUE)   # lambda penalty
)

lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
search_space_rpart <- ps(cp = p_dbl(lower = 0.001, upper = 0.1, logscale = TRUE))

lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
search_space_xgboost <- ps(
  eta = p_dbl(lower = 0.01, upper = 0.3, logscale = TRUE),   # learning rate
  max_depth = p_int(lower = 3, upper = 10),
  subsample = p_dbl(lower = 0.5, upper = 1),
  colsample_bytree = p_dbl(lower = 0.5, upper = 1),
  nrounds = p_int(lower = 50, upper = 500, logscale = TRUE)
)

lrn_nnet <- lrn("classif.nnet", predict_type = "prob", trace = FALSE)
search_space_nnet <- ps(
  size = p_int(lower = 1, upper = 10),
  decay = p_dbl(lower = 0.0001, upper = 0.1, logscale = TRUE)
)

resampling <- rsmp("cv", folds = 5)
measure <- msr("classif.auc", id = "auc")
tuner <- tnr("random_search", batch_size = 2)

tune_learner <- function(learner, search_space, term_evals = 10) {
  at <- auto_tuner(
    tuner = tuner,
    learner = learner,
    resampling = resampling,
    measure = measure,
    search_space = search_space,
    term_evals = term_evals
  )
  at$train(task_restricted)
  return(at)
}

lrn_logreg$train(task_restricted)
lrn_glmnet$train(task_restricted)
lrn_rpart$train(task_restricted)
lrn_xgboost$train(task_restricted)
lrn_nnet$train(task_restricted)
#at_glmnet <- tune_learner(lrn_glmnet, search_space_glmnet, term_evals = 10)
#at_rpart <- tune_learner(lrn_rpart, search_space_rpart, term_evals = 10)
#at_xgboost <- tune_learner(lrn_xgboost, search_space_xgboost, term_evals = 10)
#at_nnet <- tune_learner(lrn_nnet, search_space_nnet, term_evals = 10)

# test performance
learners_list <- list(
  lrn_logreg,
  lrn_glmnet,
  lrn_rpart,
  lrn_xgboost,
  lrn_nnet
)

measures <- list(
  acc = msr("classif.acc"),
  recall = msr("classif.recall"),
  spec = msr("classif.specificity"),
  auc = msr("classif.auc")
)

results <- rbindlist(lapply(learners_list, function(learner) {
  # ensure predict_type for AUC/prob-based measures
  pred <- learner$predict(test_restricted)
  scores <- sapply(measures, function(m) pred$score(m))
  data.table(
    learner = learner$id,
    class_error = 1 - scores["acc.classif.acc"],
    sensitivity = scores["recall.classif.recall"],
    specificity = scores["spec.classif.specificity"],
    auc = scores["auc.classif.auc"]
  )
}))

print(results)


# fairness - computed using nnet
ref_data <- task_train$data()[sample(task_train$nrow, 1000), ..protected_features]
ref_dt <- as.data.table(ref_data)

fair_rule_predict <- function(permitted_matrix, model = final_learner, ref_protected = ref_dt) {
  n_ref <- nrow(ref_protected)
  n_test <- nrow(permitted_matrix)
  
  preds_list <- vector("list", n_ref)
  
  for (i in 1:n_ref) {
    test_rows <- data.frame(matrix(NA, nrow = n_test, ncol = length(permitted_features) + length(protected_features)))
    colnames(test_rows) <- c(permitted_features, protected_features)
    test_rows[, permitted_features] <- permitted_matrix
    for (j in seq_along(protected_features)) {
      test_rows[, protected_features[j]] <- ref_protected[[j]][i]
    }
    preds_list[[i]] <- final_learner$predict_newdata(newdata = test_rows, predict_type = "prob")$prob[, "1"]
  }
  
  # Average over reference distribution
  rowMeans(do.call(cbind, preds_list))
}

test_all <- task_test$data(cols = task_test$feature_names)

pred_original <- final_learner$predict_newdata(test_all, predict_type = "prob")$prob[, "1"]
pred_fair <- fair_rule_predict(as.data.frame(test_permitted))
pred_restricted <- lrn_nnet$predict_newdata(test_permitted, predict_type = "prob")$prob[, "1"]

# Compute recall 
threshold <- 0.22

recall_orig <- mean(pred_original[test_true == 1] >= threshold)
recall_fair <- mean(pred_fair[test_true == 1] >= threshold)
recall_restricted <- mean(pred_restricted[test_true == 1] >= threshold)

cat("Test recall (sensitivity) for diabetes:\n")
cat(sprintf("Original model:      %.3f\n", recall_orig))
cat(sprintf("Fair rule:           %.3f\n", recall_fair))
cat(sprintf("Restricted model:    %.3f\n", recall_restricted))
auc(test_true, pred_original, positive="1")
auc(test_true, pred_fair, positive="1")
auc(test_true, pred_restricted, positive="1")
