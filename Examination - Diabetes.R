# Examination - Diabetes.R
# Authors: Therese Holmager, Arnav Tandon, Sebastian Winkelmann
# CDC Diabetes Health Indicators

library(mlr3)
library(mlr3learners)
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
task = as_task_classif(diabetes, target = "Diabetes_binary")
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

task$positive = "1"


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


