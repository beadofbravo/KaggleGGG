library(tidymodels)
library(tidyverse)
library(vroom)
library(ggmosaic)


ggg_train <- vroom("./train.csv")
ggg_test <- vroom("./test.csv")
ggg_missing <- vroom("./trainWithMissingValues.csv")


###############################
## Exploratory Data Analysis ##
###############################

ggplot(data=ggg_train, aes(x=type, y=bone_length)) +
  geom_boxplot()

ggplot(data=ggg_train) + 
  geom_mosaic(aes(x=product(color), fill=type))


############
## Recipe ##
############

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
