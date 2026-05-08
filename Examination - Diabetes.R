library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3mbo)
library(mlr3pipelines)
library(paradox)
library(future)
library(tidyverse)

future::plan("multisession") 
# future::plan("multicore")  # only works on Linux/macOS

#################################
####### Option 1: CDC Diabetes Health Indicators
#For the CDC Diabetes Health Indicators dataset, you should build classification models predicting Diabetes_binary.

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

#### 2. Choice of models
#Which models are relevant for this task? Which models did you try, which model or models did you choose to focus on, and why?

task = as_task_classif(diabetes, target = "Diabetes_binary")
set.seed(2026)
splits = partition(task, ratio = 0.8)

task_train = task$clone()$filter(splits$train)
task_test = task$clone()$filter(splits$test)

#Logistic Regression

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




#Elastic net regularization
my_elasticnet_learner = lrn(
  "classif.glmnet", # logistic regression
  standardize = FALSE,
  alpha = to_tune(0, 1),
  lambda = to_tune(p_dbl(log(1e-5), log(1e1), trafo = exp))
  # Alternatively lambda = to_tune(1e-5, 1e1, logscale=TRUE)
)

# create a graph that first dummy-encodes factors, and afterwards applies elastic net
graph_learner_elastic_net = as_learner(
  po("encode", method = "treatment", id = "binary_enc") %>>%
    po("scale") %>>%
    my_elasticnet_learner
)

set.seed(2026)
instance = tune(
  tuner = tnr("random_search", batch_size=20), ### tuning method
  #tuner = tnr("mbo"), ### tuning method
  task = task_train,
  learner = graph_learner_elastic_net,
  resampling = rsmp("cv", folds = 5), #### resampling method: 5-fold cross validation
  measures = msr("classif.ce"), #### classification error
  terminator = trm("evals", n_evals = 50) #### terminator
)

as.data.table(instance$archive$data)
instance$result_learner_param_vals
# note: lambda is stored on the internal log scale in instance$archive$data
# because we tune it through p_dbl(..., trafo = exp)

instance$result_y

# keep an untuned copy of the same graph for the auto_tuner example below
graph_learner_elastic_net_auto = graph_learner_elastic_net$clone(deep = TRUE)

# result on test set
graph_learner_elastic_net$param_set$values = instance$result_learner_param_vals
graph_learner_elastic_net$train(task_train)
graph_learner_elastic_net$predict(task_test)$score(msr("classif.ce"))


#### 3. Model performance
#How well do your models predict the desired outcome? Which performance metrics do you use, and why are they appropriate for your chosen problem?

#Logistic Regression
#Cross-Entropy (CE) = 0.1363923

#Baseline with no features
#Cross-Entropy (CE) = 0.139881

##Elastic net regularization (Best prediction)
#Cross-Entropy (CE) = 0.1357222 



  
  #### 3. Interpretability and feature importance
#Which features, or combinations of features, appear important for prediction? How do they impact the prediction? To what extent is it possible to understand or interpret the predictions made by your chosen model?
  
#Cross-Entropy: How far is the predictions of the model from the true values.
#Specificity: Ability to correctly identify true negatives. How many of the no-diabetes cases we predict truly dont have diabetes.
#Sensitivity: The ability to correctly identify true positives. How many of the diabetes cases we predict truly have diabetes.
#Accuracy: Overall correct predictions
#Precision


  #### 3.Interpretability vs. accuracy
#You should explicitly discuss the trade-off between interpretability and predictive accuracy.

#### 3. Sensitive or protected features
#You should explicitly consider the role of potentially sensitive features in your chosen dataset.
#For the CDC Diabetes Health Indicators dataset, this this could for example be a variables such as BMI.

#In addition to your main final model, you should report a second model that does not make use of chosen sensitive variables, neither directly nor indirectly. 
#You should explain how you identified and handled such variables, and you should compare the original model and the restricted model. In particular, you should discuss what changed.

