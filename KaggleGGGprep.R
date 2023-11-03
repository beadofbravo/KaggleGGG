library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(ggmosaic)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kernlab)
library(themis)


ggg_train <- vroom("./train.csv") %>%
  select(-c(id))
ggg_test <- vroom("./test.csv")
#ggg_missing <- vroom("./trainWithMissingValues.csv")

my_recipe <- recipe(type ~., data = ggg_train) %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()

###############################
## Exploratory Data Analysis ##
###############################

ggplot(data=ggg_train, aes(x=type, y=bone_length)) +
  geom_boxplot()

ggplot(data=ggg_train) + 
  geom_mosaic(aes(x=product(color), fill=type))


#####################
## Recipe  Missing ##
#####################

my_recipe_ggg <- recipe(type ~., data = ggg_missing) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul, color, type), neighbors = 10) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, type, bone_length), neighbors = 10) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color, type, bone_length, rotting_flesh), neighbors = 10) %>%
  prep()

my_recipe_ggg_bag <- recipe(type ~., data = ggg_missing) %>%
  step_impute_bag(bone_length, impute_with = imp_vars(has_soul, color, type), trees = 500) %>%
  step_impute_bag(rotting_flesh, impute_with = imp_vars(has_soul, color, type, bone_length), trees = 500) %>%
  step_impute_bag(hair_length, impute_with = imp_vars(has_soul, color, type, bone_length, rotting_flesh), trees = 500) %>%
  prep()

my_recipe_ggg_mean <- recipe(type ~., data = ggg_missing) %>%
  step_impute_mean(bone_length) %>%
  step_impute_mean(rotting_flesh) %>%
  step_impute_mean(hair_length) %>%
  prep()

my_recipe_ggg_median <- recipe(type ~., data = ggg_missing) %>%
  step_impute_median(bone_length) %>%
  step_impute_median(rotting_flesh) %>%
  step_impute_median(hair_length) %>%
  prep()

my_recipe_ggg_linear <- recipe(type ~., data = ggg_missing) %>%
  step_impute_linear(rotting_flesh, impute_with = imp_vars(has_soul, color, type)) %>%
  step_impute_linear(bone_length, impute_with = imp_vars(has_soul, color, type, rotting_flesh)) %>%
  step_impute_linear(hair_length, impute_with = imp_vars(has_soul, color, type, bone_length, rotting_flesh)) %>%
  prep()

my_recipe_ggg_linear_order <- recipe(type ~., data = ggg_missing) %>%
  step_impute_linear(hair_length, impute_with = imp_vars(has_soul, color, type)) %>%
  step_impute_linear(rotting_flesh, impute_with = imp_vars(has_soul, color, type, hair_length)) %>%
  step_impute_linear(bone_length, impute_with = imp_vars(has_soul, color, type, hair_length, rotting_flesh)) %>%
  prep()

prep <- prep(my_recipe_ggg_linear_order)
bake <- bake(prep, new_data = NULL)

rmse_vec(ggg_train[is.na(ggg_missing)], bake[is.na(ggg_missing)])

sum(is.na(ggg_missing$bone_length))
sum(is.na(ggg_missing$rotting_flesh))
sum(is.na(ggg_missing$hair_length))



#################
## Naive Bayes ##
#################

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid_nb <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

nb_results_tune <- nb_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_nb,
            metrics=metric_set(accuracy))

bestTune <- nb_results_tune %>%
  select_best("accuracy")


final_wf_nb <- nb_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

nb_predictions <- final_wf_nb %>%
  predict(new_data = ggg_test, type = "class")

ggg_predictions_nb <- nb_predictions %>%
  bind_cols(., ggg_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)


vroom_write(x=ggg_predictions_nb, file="./nb.csv", delim=",")

####################
## Random Forests ##
####################

my_mod_crf <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

class_reg_tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_crf)

tuning_grid_crf <- grid_regular(min_n(),
                                mtry(range = c(1, 10)),
                                levels = 5)


folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

CV_results_crf <- class_reg_tree_wf %>%
  tune_grid(resamples = folds, 
            grid = tuning_grid_crf,
            metrics=metric_set(roc_auc))

bestTune <- CV_results_crf %>%
  select_best("roc_auc")

final_wf_crf <- class_reg_tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

plr_predictions <- predict(final_wf_crf, new_data = ggg_test, type = "prob")


amazon_predictions_plr <- plr_predictions %>%
  bind_cols(., empl_access_test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)


vroom_write(x=amazon_predictions_plr, file="./crfbd.csv", delim=",")



#############################
## Support Vector Machines ##
#############################

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

tuning_grid_svm <- grid_regular(cost(),
                                rbf_sigma(),
                                levels = 5)


folds <- vfold_cv(ggg_train, v = 5, repeats = 1)


svm_results_tune <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_svm,
            metrics=metric_set(accuracy))


bestTune <- svm_results_tune %>%
  select_best("accuracy")


final_wf_svm <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

svm_predictions <- final_wf_svm %>%
  predict(new_data = ggg_test, type = "class")

amazon_predictions_svm <- svm_predictions %>%
  bind_cols(., ggg_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)


vroom_write(x=amazon_predictions_svm, file="./svmradbd.csv", delim=",")


#########################
## K Nearest Neighbors ##
#########################

knn_mod <- nearest_neighbor(neighbors = tune()) %>%
  set_mode('classification') %>%
  set_engine('kknn')


knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_mod)


tune_grid <- grid_regular(neighbors(), levels = 10)


CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tune_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")


final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)


test_preds <- final_wf %>%
  predict(new_data=ggg_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., ggg_test) %>%
  select(id, ACTION)


vroom_write(x=test_preds, file="./knnbd.csv", delim=",")

################



