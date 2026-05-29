#### Option 1: CDC Diabetes Health Indicators
#For the CDC Diabetes Health Indicators dataset, you should build classification models predicting Diabetes_binary.


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

future::plan("multisession") 
# future::plan("multicore")  # only works on Linux/macOS

#Import dataset
diabetes <- read.csv("C:/Users/THOL0099/Desktop/Diabetes.csv")
#The dataset has 253680 observations and 22 variables


#### 1. Pre-processing. Do you need to perform any pre-processing?

#I use table to check if all values match the variable description, e.g. binary variables being 0 or 1, and see missing values.
table(diabetes$Diabetes_binary, useNA = "ifany")
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
24206/253680
#24206 duplicated values (9.5% of all observations)
#We chose not to delete duplicated rows, as we believe that more than one person with identical values for the 22 variables with few categories may be possible.


#### 2. Choice of models
#Which models are relevant for this task? Which models did you try, which model or models did you choose to focus on, and why?

#We start by define task in the dataset and splitting it (80% training 20% test)
task = as_task_classif(diabetes, target = "Diabetes_binary", positive = "1")
set.seed(2026)
splits = partition(task, ratio = 0.8)

task_train = task$clone()$filter(splits$train)
task_test = task$clone()$filter(splits$test)


#Logistic Regression. Logistic regression can model probabilities for a binary outcome (diabetes yes/no) 

my_lr_learner = lrn("classif.log_reg")

graph_learner_lr = as_learner(
  po("encode", method = "treatment", id = "binary_enc") %>>%
    po("scale") %>>%
    my_lr_learner
)

graph_learner_lr$train(task_train)
graph_learner_lr$predict(task_test)$score(msr("classif.ce"))



#Baseline with no features

baseline <- lrn("classif.featureless")$train(task_train)
baseline$predict(task_test)$score(msr("classif.ce"))




#Elastic net. Elastic net combines Lasso (L1) and Rigde (L2) regularization in a model
#First we create a learner.
my_elasticnet_learner = lrn(
  "classif.glmnet", 
  standardize = FALSE,
  alpha = to_tune(0, 1),
  lambda = to_tune(p_dbl(log(1e-5), log(1e1), trafo = exp))
  # Alternatively lambda = to_tune(1e-5, 1e1, logscale=TRUE)
)

#We create a graph that converts factor variables into dummy variables, standardizes predictors, and afterwards applies elastic net
graph_learner_elastic_net = as_learner(
  po("encode", method = "treatment", id = "binary_enc") %>>%
    po("scale") %>>%
    my_elasticnet_learner
)


#Now we apply 5-fold cross-validation (train on 4 folds, validate on 1 fold, repeat 5 times). 
#Cross-validation is used to reduce overfitting.
set.seed(2026)
instance = tune(
  tuner = tnr("random_search", batch_size=20), 
  task = task_train,
  learner = graph_learner_elastic_net,
  resampling = rsmp("cv", folds = 5), 
  measures = msr("classif.ce"),
  terminator = trm("evals", n_evals = 50) 
)

as.data.table(instance$archive$data)
instance$result_learner_param_vals

instance$result_y

# keep an untuned copy of the same graph for the auto_tuner
graph_learner_elastic_net_auto = graph_learner_elastic_net$clone(deep = TRUE)

# result on test set
graph_learner_elastic_net$param_set$values = instance$result_learner_param_vals
graph_learner_elastic_net$train(task_train)
graph_learner_elastic_net$predict(task_test)$score(msr("classif.ce"))




#Decision tree
set.seed(123)

# load data and define task
task = as_task_classif(diabetes, target = "Diabetes_binary")

# Diabetes = 1 is the outcome of interest.
positive_class = "1"
task$positive = positive_class

#Create a graph that dummy-encodes categorical features and then applies learner
categorical_selector = selector_type(c("factor", "ordered"))

make_encoded_learner = function(learner) {
  as_learner(
    po("encode", method = "treatment",
       affect_columns = categorical_selector, id = "binary_enc") %>>%
      learner
  )
}

# and visualize a large CART tree
my_cart_learner = lrn("classif.rpart", cp = 0, predict_type = "prob")

full_tree = make_encoded_learner(my_cart_learner)
full_tree$train(task)

cart_trained <- full_tree$model$classif.rpart$model
plot(cart_trained, compress = TRUE, margin = 0.1)
text(cart_trained, use.n = TRUE, cex = 0.4) # Adjust cex (font size) as needed


# Full tree CE
rr_full = resample(
  task,
  full_tree,
  rsmp("cv", folds = 5)
)

rr_full$aggregate(msr("classif.ce"))

#Cross-validation path for pruning parameter (weakest-link pruning)
my_cart_learner_cv = lrn("classif.rpart",
                         cp = 0,
                         xval = 5,
                         predict_type = "prob")

graph_learner_cart_cv = make_encoded_learner(my_cart_learner_cv)
graph_learner_cart_cv$train(task)

cart_trained_cv <- graph_learner_cart_cv$model$classif.rpart$model

rpart::plotcp(cart_trained_cv)
rpart::printcp(cart_trained_cv)

#Select cp by the one-standard-error rule and prune
cp_table = cart_trained_cv$cptable
min_row = which.min(cp_table[, "xerror"])
one_se_threshold = cp_table[min_row, "xerror"] + cp_table[min_row, "xstd"]

# rpart orders the table from simpler to larger trees, so the first row within
# the one-SE threshold gives the simplest acceptable tree.
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


# Pruned tree CE for weakest-link pruning
rr_pruned = resample(
  task,
  pruned_tree,
  rsmp("cv", folds = 5)
)

rr_pruned$aggregate(msr("classif.ce"))



#Random Forest
rf_learner = lrn(
  "classif.ranger",
  predict_type = "prob"
)

rf_model = make_encoded_learner(rf_learner)

rf_model$train(task)

#CE for random forest
rr_rf = resample(
  task,
  rf_model,
  rsmp("cv", folds = 5)
)

rr_rf$aggregate(msr("classif.ce"))



#XGboost
xgb_learner = lrn(
  "classif.xgboost",
  predict_type = "prob"
)

xgb_model = make_encoded_learner(xgb_learner)

xgb_model$train(task)

#CE for XGboost
rr_xgb = resample(
  task,
  xgb_model,
  rsmp("cv", folds = 5)
)

rr_xgb$aggregate(msr("classif.ce"))



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


#### 3. Model performance
#How well do your models predict the desired outcome? Which performance metrics do you use, and why are they appropriate for your chosen problem?

#We used classification error (Number of incorrect predictions / Total number of predictions). 

#Logistic Regression
#Classification error (CE) = 0.1363923

#Baseline with no features
#Classification error (CE) = 0.139881

##Elastic net regularization 
#Classification error (CE) = 0.1357222 

##Full decision tree 
#Classification error (CE) =  0.1558341

##Pruned decision tree (weakest-link pruning)
#Classification error (CE) = 0.1346421

##Decision tree (Random forest)
#Classification error (CE) = 0.133763

##Decision tree (XGboost)
#Classification error (CE) = 0.1416391


#### 4. Interpretability and feature importance
#Which features, or combinations of features, appear important for prediction? How do they impact the prediction? 
#To what extent is it possible to understand or interpret the predictions made by your chosen model?

#Classification error: How far is the predictions of the model from the true values.
#Specificity: Ability to correctly identify true negatives. How many of the no-diabetes cases we predict truly dont have diabetes.
#Sensitivity: The ability to correctly identify true positives. How many of the diabetes cases we predict truly have diabetes.
#Accuracy: Overall correct predictions
#Precision


#### 5.Interpretability vs. accuracy
#You should explicitly discuss the trade-off between interpretability and predictive accuracy.

#The simpler model (logistic reg) is more interpretable but less accurate than e.g. elastic net regularization.
#Show examples with model naive model, logistic etc.


#### 6. Sensitive or protected features
#You should explicitly consider the role of potentially sensitive features in your chosen dataset.
#For the CDC Diabetes Health Indicators dataset, this this could for example be a variables such as BMI.
#In addition to your main final model, you should report a second model that does not make use of chosen sensitive variables, neither directly nor indirectly. 
#You should explain how you identified and handled such variables, and you should compare the original model and the restricted model. In particular, you should discuss what changed.


#The most relevant variable are sensitive
#Vizualisation for sensitive variables and non-sensitive for comparison

#Sensitive variables can be identified by allready established correlations e.g. increased risk of diabetes with increasing BMI and also by modelling the data.
#compare prediction of models without sensitive variables. We would expect less accuracy using the same models without the the sensitive variables.
