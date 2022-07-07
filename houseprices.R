# ******************************************************************************
# 
# Comparative analysis of imputation of real estate transaction data 
# using neural networks, random forest, kriging and inverse-distance weighting
#
# Author: Seong-Yun Hong (syhong@khu.ac.kr), Jeong-Hyeon Kim (zmsdkdle@khu.ac.kr)
# Date: 02 January 2022
#
# ******************************************************************************

library(keras)
library(mlbench)
library(psych)
library(dplyr)
library(magrittr)
library(neuralnet)
library(caret)
library(MLmetrics)
library(Metrics)
library(rgdal)
library(car)
library(gstat)
library(MLmetrics)
library(Metrics)
library(caret)
library(gridExtra)
library(ranger)
options(scipen = 20)

# ******************************************************************************
# 
# ---- load and preprocessing data ----
# 
# ******************************************************************************

# flr_area = Gross Floor Area
# bcr = Building coverage ratio
# far = Floor area ratio
varnames <- c("bldg_area", "yr_built", "flr_area", "site_area", "height", 
              "bcr", "far", "x", "y", "district", "yr", "month", "net_area",
              "price", "flr_lv", "type")

## load data
train <- read.csv("data/train.csv", header = TRUE)
test <- read.csv("data/test.csv", header = TRUE)

# remove ID
train <- train[, -1]
test <- test[, -1]

## Training data
train_y <- train$price / train$net_area # price per net area
train_x <- train[,c(-13, -14)]
train_x$type <- as.factor(train_x$type)
train_x$district <- as.factor(train_x$district)

bldg <- dummyVars(~., data = train_x)
train_x <- predict(bldg, newdata = train_x)
train_x_std <- scale(train_x, colMeans(train_x), apply(train_x, 2, sd))

# Test data
test_y <- test$price / test$net_area
test_x <- test[, c(-13 ,-14)]
test_x$type <- as.factor(test_x$type)
test_x$district <- as.factor(test_x$district)

bldg <- dummyVars(~., data = test_x)
test_x <- predict(bldg, newdata = test_x)
test_x_std <- scale(test_x, colMeans(train_x), apply(train_x, 2, sd))

# ******************************************************************************
# 
# ---- IDW ----
# 
# ******************************************************************************

## load shp files
train_shp <- readOGR("data/train.shp")
test_shp <- readOGR("data/test.shp")

train_std <- cbind(train_x_std, train_y)
colnames(train_std)[41] <- "price_per_netarea"
train_shp@data <- data.frame(train_std)

test_std <- cbind(test_x_std, test_y)
colnames(test_std)[41] <- "price_per_netarea"
test_shp@data <- data.frame(test_std)

## calibrate zeridist point data
nrow(train_shp@data)
dist_zero <- zerodist(train_shp)
unique(dist_zero[, 1])

set.seed(1)
calibration <- rnorm(n = length(unique(dist_zero[, 1])), mean = 0.000001, sd = 0.0000001)

train_shp@coords[unique(dist_zero[, 1]), 1] <- train_shp@coords[unique(dist_zero[, 1]), 1] + calibration

# re-calibration
dist_zero <- zerodist(train_shp)
set.seed(1)
calibration <- rnorm(n = length(unique(dist_zero[, 1])), mean = 0.000001, sd = 0.0000001)
train_shp@coords[unique(dist_zero[, 1]), 1] <- train_shp@coords[unique(dist_zero[, 1]), 1] + calibration

zerodist(train_shp)



## IDW (idw = 1, 2, 4, 6, 8, 10)
idw_p <- seq(2, 8, 0.1)
idw_rmse <- c()

for (i in idw_p) {
  print(paste("idp = ", i))
  
  fit <- idw(price_per_netarea ~ 1, train_shp, test_shp, idp = i)
  fit_rmse <- RMSE(fit@data$var1.pred, test_shp@data$price)
  idw_rmse <- c(idw_rmse, fit_rmse)
}

idw_p[order(idw_rmse)] # 2.2 일때 가장 rmse가 낮음
plot(idw_p, idw_rmse, type='l', main = 'RMSE with different distance decaying parameters',
     xlab = 'Decay Factor (p)', ylab='RMSE')

# best IDW
best_idw <- fit <- idw(price_per_netarea ~ 1, train_shp, test_shp, 
                       idp = 2.2)


# ******************************************************************************
# 
# ---- Kriging ----
# 
# ******************************************************************************
l
## create semivariogram
v <- variogram(price_per_netarea ~ 1, train_shp, width=500, cutoff=10000)
plot(v)


## variogram fitting
## shp model
v.fit_sph1 <- fit.variogram(v, model = vgm(NA, 'Sph', range=1000, nugget=NA),
                            fit.sills = T, fit.ranges = F)
v.fit_sph2 <- fit.variogram(v, model = vgm(NA, 'Sph', range=3000, nugget=NA),
                            fit.sills = T, fit.ranges = F)
v.fit_sph3 <- fit.variogram(v, model = vgm(NA, 'Sph', range=5000, nugget=NA),
                            fit.sills = T, fit.ranges = F)
v.fit_sph4 <- fit.variogram(v, model = vgm(NA, 'Sph', range=7000, nugget=NA),
                            fit.sills = T, fit.ranges = F)

par(mfrow=c(1, 3))

line1 = variogramLine(v.fit_sph1, maxdist = 15000)
line2 = variogramLine(v.fit_sph2, maxdist = 15000)
line3 = variogramLine(v.fit_sph3, maxdist = 15000)
line4 = variogramLine(v.fit_sph4, maxdist = 15000)

plot(line1, type='l', xlim = c(0, 10000) ,ylim = c(20000,80000), lty='solid', col='black', lwd=1.5,
     xlab = 'Distance', ylab = 'Semivariance', xaxs = "i", axes=FALSE, cex.lab=1.2)
axis(1, xaxs='i', at=c(5000, 10000)); axis(2, at=c(20000, 40000, 60000, 80000)); box()
lines(line2, lty='dashed', col='black', lwd=1.5)
lines(line3, lty='dotted', col='black', lwd=1.5)
lines(line4, lty='dotdash', col='black', lwd=1.5)
points(v$dist, v$gamma, col='Blue', cex=1, lwd=1.5)
legend(300, 82000, legend=c("model 1", "model 2", "model 3", "model 4", "Experimental variogram data"), 
       lty=c("solid", "dashed", "dotted", "dotdash", NA), 
       pch=c(NA, NA, NA, NA, 21), 
       col=c('Black','Black', 'Black', 'black','Blue'), cex=1.2)

krige_sph1 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_sph1,
                    nmax = 50)
krige_sph2 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_sph2,
                    nmax = 50)
krige_sph3 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_sph3,
                    nmax = 50)
krige_sph4 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_sph4,
                    nmax = 50)

RMSE(krige_sph1@data$var1.pred, test_shp@data$price)
RMSE(krige_sph2@data$var1.pred, test_shp@data$price)
RMSE(krige_sph3@data$var1.pred, test_shp@data$price)
RMSE(krige_sph4@data$var1.pred, test_shp@data$price)

## exp model
v.fit_exp1 <- fit.variogram(v, model = vgm(NA, 'Exp', range=1000, nugget=NA),
                            fit.sills = TRUE, fit.ranges = FALSE)
v.fit_exp2 <- fit.variogram(v, model = vgm(NA, 'Exp', range=3000, nugget=NA),
                            fit.sills = TRUE, fit.ranges = FALSE)
v.fit_exp3 <- fit.variogram(v, model = vgm(NA, 'Exp', range=5000, nugget=NA),
                            fit.sills = TRUE, fit.ranges = FALSE)
v.fit_exp4 <- fit.variogram(v, model = vgm(NA, 'Exp', range=8000, nugget=NA),
                            fit.sills = TRUE, fit.ranges = FALSE)

line1 = variogramLine(v.fit_exp1, maxdist = 15000)
line2 = variogramLine(v.fit_exp2, maxdist = 15000)
line3 = variogramLine(v.fit_exp3, maxdist = 15000)
line4 = variogramLine(v.fit_exp4, maxdist = 15000)
plot(line1, type='l', xlim = c(0, 10000) ,ylim = c(20000,80000), lty='solid', col='black', lwd=1.5,
     xlab = 'Distance', ylab = 'Semivariance', xaxs = "i", axes=FALSE, cex.lab=1.2)
axis(1, xaxs='i', at=c(5000, 10000)); axis(2, at=c(20000, 40000, 60000, 80000)); box()
lines(line2, lty='dashed', col='black', lwd=1.5)
lines(line3, lty='dotted', col='black', lwd=1.5)
lines(line4, lty='dotdash', col='black', lwd=1.5)
points(v$dist, v$gamma, col='Blue', cex=1, lwd=1.5)
legend(300, 82000, legend=c("model 5", "model 6", "model 7", "model 8", "Experimental variogram data"), 
       lty=c("solid", "dashed", "dotted", "dotdash", NA), 
       pch=c(NA, NA, NA, NA, 21), 
       col=c('Black','Black', 'Black', 'black','Blue'), cex=1.2)

krige_exp1 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_exp1,
                    nmax = 50)
krige_exp2 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_exp2,
                    nmax = 50)
krige_exp3 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_exp3,
                    nmax = 50)
krige_exp4 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_exp4,
                    nmax = 50)

RMSE(krige_exp1@data$var1.pred, test_shp@data$price)
RMSE(krige_exp2@data$var1.pred, test_shp@data$price)
RMSE(krige_exp3@data$var1.pred, test_shp@data$price)
RMSE(krige_exp4@data$var1.pred, test_shp@data$price)

## gau model
v.fit_gau1 <- fit.variogram(v, model = vgm(NA, 'Gau', range=1000, nugget=NA),
                            fit.sills = TRUE, fit.ranges = FALSE)
v.fit_gau2 <- fit.variogram(v, model = vgm(NA, 'Gau', range=3000, nugget=NA),
                            fit.sills = TRUE, fit.ranges = FALSE)
v.fit_gau3 <- fit.variogram(v, model = vgm(NA, 'Gau', range=5000, nugget=NA),
                            fit.sills = TRUE, fit.ranges = FALSE)
v.fit_gau4 <- fit.variogram(v, model = vgm(NA, 'Gau', range=8000, nugget=NA),
                            fit.sills = TRUE, fit.ranges = FALSE)

line1 = variogramLine(v.fit_gau1, maxdist = 15000)
line2 = variogramLine(v.fit_gau2, maxdist = 15000)
line3 = variogramLine(v.fit_gau3, maxdist = 15000)
line4 = variogramLine(v.fit_gau4, maxdist = 15000)
plot(line1, type='l', xlim = c(0, 10000) ,ylim = c(20000,80000), lty='solid', col='black', lwd=1.5,
     xlab = 'Distance', ylab = 'Semivariance', xaxs = "i", axes=FALSE, cex.lab=1.2)
axis(1, xaxs='i', at=c(5000, 10000)); axis(2, at=c(20000, 40000, 60000, 80000)); box()
lines(line2, lty='dashed', col='black', lwd=1.5)
lines(line3, lty='dotted', col='black', lwd=1.5)
lines(line4, lty='dotdash', col='black', lwd=1.5)
points(v$dist, v$gamma, col='Blue', cex=1, lwd=1.5)
legend(300, 82000, legend=c("model 9", "model 10", "model 11", "model 12", "Experimental variogram data"), 
       lty=c("solid", "dashed", "dotted", "dotdash", NA), 
       pch=c(NA, NA, NA, NA, 21), 
       col=c('Black','Black', 'Black', 'black','Blue'), cex=1.2)

krige_gau1 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_gau1,
                    nmax = 50)
krige_gau2 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_gau2,
                    nmax = 50)
krige_gau3 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_gau3,
                    nmax = 50)
krige_gau4 <- krige(price_per_netarea ~ 1, 
                    train_shp, 
                    test_shp,
                    model = v.fit_gau4,
                    nmax = 50)

RMSE(krige_gau1@data$var1.pred, test_shp@data$price)
RMSE(krige_gau2@data$var1.pred, test_shp@data$price)
RMSE(krige_gau3@data$var1.pred, test_shp@data$price)
RMSE(krige_gau4@data$var1.pred, test_shp@data$price)

## best kriging
best_kriging <- krige_sph1


# ******************************************************************************
# 
# ---- Random Forest ----
# 
# ******************************************************************************
train_new <- cbind(train_x_std, train_y)
test_new <- cbind(test_x_std, test_y)

## model hyperparameter
model_grid <- expand.grid(mtry = c(2, 4, 8, 16), splitrule = "variance",
                          min.node.size = 5)

## create model
for (i in 1:6) {
  print(paste("# trees:", 50 * i))
  
  set.seed(1)
  fit <- train(train_y ~ .,
               data = train_new,
               method = "ranger",
               trControl = trainControl(method = "cv", number = 10, 
                                        allowParallel = TRUE, verbose = TRUE),
               tuneGrid = model_grid,
               importance = "permutation",
               num.trees = 50 * i)
  
  assign(paste0("rf_tree_", 50 * i), fit)
}


## predict
pred_tree_50 <- predict(rf_tree_50, test_new)
pred_tree_100 <- predict(rf_tree_100, test_new)
pred_tree_150 <- predict(rf_tree_150, test_new)
pred_tree_200 <- predict(rf_tree_200, test_new)
pred_tree_250 <- predict(rf_tree_250, test_new)
pred_tree_300 <- predict(rf_tree_300, test_new)

RMSE(pred_tree_50, test_y)
RMSE(pred_tree_100, test_y)
RMSE(pred_tree_150, test_y)
RMSE(pred_tree_200, test_y)
RMSE(pred_tree_250, test_y)
RMSE(pred_tree_300, test_y)

RMSE(pred_tree_300, test_y) # 80.87895
MAE(pred_tree_300, test_y) # 48.18974
mape(pred_tree_300, test_y) # 0.08954253
mase(pred_tree_300, test_y) # 0.226314


## variable importace 
imp_variable <- data.frame(type = rownames(varImp(rf_tree_300)$importance), 
                           importance = varImp(rf_tree_300)$importance$Overall)
imp_variable <- imp_variable %>% arrange(desc(importance))
imp_var_top15 <- imp_variable[1:15, ]
imp_var_top15
imp_name <- c("Y-coordinate", "X-coordinate", "Date of Approval for Use", 'height',
              "Gu_Gangnam", "Land area", "Floor", "Floor area ratio", "Building coverage ratio",
              "Total Area", "Contact Year", "Building Area", "Gu_Seocho", 
              "Type Apartments", "Type Multi")
par(oma=c(0, 6, 0, 0))
b_plot <- barplot(imp_var_top15$importance[seq(15, 1)], 
                  names.arg = imp_name[seq(15, 1)], # x축 이름
                  horiz = TRUE, # 수직
                  cex.names = 1,
                  las = 1,
                  col = 'grey',
                  border = "white",
                  main = "Variable Importance")
abline(v = c(20, 40, 60, 80), lty = 2, col = "grey80")
box(col = "black")
text(x = imp_var_top15$importance[seq(15, 1)] - 2,
     y = b_plot,
     label = round(imp_var_top15$importance[seq(15, 1)], 1),
     cex = 0.8)



# ******************************************************************************
# 
# ---- ANN ----
# 
# ******************************************************************************

# create custom rmse metric
custom_rmse <- function(y_true, y_pred){
  K        <- backend()
  # calculate the metric
  temp <- K$sqrt(K$mean(K$square(y_pred -y_true))) 
  temp
}
metric_rmse <- custom_metric("RMSE", function(y_true, y_pred) {
  custom_rmse(y_true, y_pred)
})


## model 1 #####################################################################
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 32, activation = 'elu', kernel_initializer = "he_normal",
              kernel_regularizer = regularizer_l2(), input_shape = c(40), name = 'input') %>%
  layer_dense(units = 16, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden1') %>%
  layer_dense(units = 1, name = 'output')
summary(model)

model %>% compile(loss = 'mse', optimizer = optimizer_adam(), 
                  metrics = metric_rmse)
set.seed(1)

ann1 <- model %>% 
  fit(train_x_std, train_y, epochs = 300, batch_size = 1000, 
      validation_split = 0.2)
ann1
# val_RMSE: 131.5

value1 <- model %>% 
  predict_on_batch(test_x_std)

RMSE(value1, test_y) 
# test_RMSE: 131.6472

## model2 #####################################################################
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 64, activation = 'elu', kernel_initializer = "he_normal",
              kernel_regularizer = regularizer_l2(), input_shape = c(40), name = 'input') %>%
  layer_dense(units = 32, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden1') %>%
  layer_dense(units = 16, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden2') %>%
  layer_dense(units = 1, name = 'output')
summary(model)

model %>% compile(loss = 'mse', optimizer = optimizer_adam(), 
                  metrics = metric_rmse)

ann2 <- model %>% 
  fit(train_x_std, train_y, epochs = 300, batch_size = 1000, 
      validation_split = 0.2)
ann2
# val_RMSE: 121.4

value2 <- model %>% 
  predict_on_batch(test_x_std)

RMSE(value2, test_y) # test_RMSE: 121.7526

## model3 #####################################################################
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 128, activation = 'elu', kernel_initializer = "he_normal",
              kernel_regularizer = regularizer_l2(), input_shape = c(40), name = 'input') %>%
  layer_dense(units = 64, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden1') %>%
  layer_dense(units = 32, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden2') %>%
  layer_dense(units = 16, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden3') %>%
  layer_dense(units = 1, name = 'output')
summary(model)

model %>% compile(loss = 'mse', optimizer = optimizer_adam(), 
                  metrics = metric_rmse)

ann3 <- model %>% 
  fit(train_x_std, train_y, epochs = 300, batch_size = 1000, 
      validation_split = 0.2)
ann3
# val_RMSE: 114.3

value3 <- model %>% 
  predict_on_batch(test_x_std)

RMSE(value3, test_y) # test_RMSE: 114.2375

## model4 #####################################################################
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'elu', kernel_initializer = "he_normal",
              kernel_regularizer = regularizer_l2(), input_shape = c(40), name = 'input') %>%
  layer_dense(units = 128, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden1') %>%
  layer_dense(units = 64, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden2') %>%
  layer_dense(units = 32, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden3') %>%
  layer_dense(units = 16, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden4') %>%
  layer_dense(units = 1, name = 'output')
summary(model)

model %>% compile(loss = 'mse', optimizer = optimizer_adam(), 
                  metrics = metric_rmse)

ann4 <- model %>% 
  fit(train_x_std, train_y, epochs = 300, batch_size = 1000, 
      validation_split = 0.2)
ann4
# val_RMSE: 106.1

value4 <- model %>% 
  predict_on_batch(test_x_std)

RMSE(value4, test_y) # test_RMSE: 105.8632

## model5 #####################################################################
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 512, activation = 'elu', kernel_initializer = "he_normal",
              kernel_regularizer = regularizer_l2(), input_shape = c(40), name = 'input') %>%
  layer_dense(units = 256, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden1') %>%
  layer_dense(units = 128, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden2') %>%
  layer_dense(units = 64, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden3') %>%
  layer_dense(units = 32, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden4') %>%
  layer_dense(units = 16, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden5') %>%
  layer_dense(units = 1, name = 'output')
summary(model)

model %>% compile(loss = 'mse', optimizer = optimizer_adam(), 
                  metrics = metric_rmse)

ann5 <- model %>% 
  fit(train_x_std, train_y, epochs = 300, batch_size = 1000, 
      validation_split = 0.2)
ann5
# val_RMSE: 101.3

value5 <- model %>% 
  predict_on_batch(test_x_std)

RMSE(value5, test_y) # test_RMSE: 100.0775

## model6 #####################################################################
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 1024, activation = 'elu', kernel_initializer = "he_normal",
              kernel_regularizer = regularizer_l2(), input_shape = c(40), name = 'input') %>%
  layer_dense(units = 512, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden1') %>%
  layer_dense(units = 256, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden2') %>%
  layer_dense(units = 128, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden3') %>%
  layer_dense(units = 64, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden4') %>%
  layer_dense(units = 32, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden5') %>%
  layer_dense(units = 16, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden6') %>%
  layer_dense(units = 1, name = 'output')
summary(model)

model %>% compile(loss = 'mse', optimizer = optimizer_adam(), 
                  metrics = metric_rmse)

ann6 <- model %>% 
  fit(train_x_std, train_y, epochs = 300, batch_size = 1000, 
      validation_split = 0.2)
ann6
# val_RMSE: 101.3

value6 <- model %>% 
  predict_on_batch(test_x_std)

RMSE(value6, test_y) # test_RMSE: 100.6292

## model7 #####################################################################
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 2048, activation = 'elu', kernel_initializer = "he_normal",
              kernel_regularizer = regularizer_l2(), input_shape = c(40), name = 'input') %>%
  layer_dense(units = 1024, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden1') %>%
  layer_dense(units = 512, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden2') %>%
  layer_dense(units = 256, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden3') %>%
  layer_dense(units = 128, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden4') %>%
  layer_dense(units = 64, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden5') %>%
  layer_dense(units = 32, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden6') %>%
  layer_dense(units = 16, activation = 'elu', kernel_initializer = "he_normal", 
              kernel_regularizer = regularizer_l2(), name = 'hidden7') %>%
  layer_dense(units = 1, name = 'output')
summary(model)


model %>% compile(loss = 'mse', optimizer = optimizer_adam(), 
                  metrics = metric_rmse)

ann7 <- model %>% 
  fit(train_x_std, train_y, epochs = 300, batch_size = 1000, 
      validation_split = 0.2)
ann7
# val_RMSE: 102.3

value7 <- model %>% 
  predict_on_batch(test_x_std)

RMSE(value7, test_y) # test_RMSE: 102.5375

# ann_value <- value5

## best ANN #####################################################################
set.seed(1)
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(40), name = 'input') %>%
  layer_dense(units = 256, activation = 'relu', name = 'hidden1') %>%
  layer_dense(units = 128, activation = 'relu', name = 'hidden2') %>%
  layer_dense(units = 128, activation = 'relu', name = 'hidden3') %>%
  layer_dense(units = 1, name = 'output')
summary(model)


model %>% compile(loss = 'mse', optimizer = optimizer_adam(), 
                  metrics = metric_rmse)

best_ann <- model %>% 
  fit(train_x_std, train_y, epochs = 150, batch_size = 64, 
      validation_split = 0.2)

ann_value <- model %>% 
  predict_on_batch(test_x_std)

RMSE(ann_value, test_y) # test_RMSE: 102.3457

# ******************************************************************************
# 
# ---- Extracting predict value ----
# 
# ******************************************************************************

idw_value <- best_idw@data$var1.pred
kriging_value <- best_kriging$var1.pred
# rf_value <- pred_tree_300

write.csv(idw_value, 'idw_value.csv')
write.csv(kriging_value, 'kriging_value.csv')
# write.csv(rf_value, 'rf_value.csv')
write.csv(ann_value, 'ann_value.csv')

idw_value <- read.csv('idw_value.csv')[, 2]
kriging_value <- read.csv('kriging_value.csv')[, 2]
rf_value <- read.csv('rf_value.csv')[, 2]
ann_value <- read.csv('ann_value.csv')[, 2]
observed <- test_y

pred_value <- cbind(observed, idw_value, kriging_value, rf_value, ann_value)
colnames(pred_value) <- c('observed', 'idw', 'kriging', 'rf', 'ann')
head(pred_value)

write.csv(pred_value, 'pred_value.csv')
