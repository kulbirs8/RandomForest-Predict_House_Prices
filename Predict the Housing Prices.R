### Project 1: Predict the Housing Prices in Ames



# Libraries to use

library(dplyr)
library(tidyr)
library(reshape2)
library(caret)
library(randomForest)
library(xgboost)
library(gbm)
library(glmnet)
library(mgcv)
library(gam)
library(kernlab)
library(e1071)

# Load training data
train.org <- read.csv("train.csv")

#summary(train.org)

####################
## 1. Preprocess training data
####################
train1 = train.org

# --- 1.1 Handling missing values ---
#missing.n = sapply(names(train1),
#                    function(x)
#                      length(which(is.na(train1[, x]))))
#which(missing.n != 0)  # 60th col: Garage_Yr_Blt
#id = which(is.na(train1$Garage_Yr_Blt))
#length(id)

# Replacing missing values on variable "Garage_Yr_Blt" with 0
train1[is.na(train1)] = 0

# --- 1.2 Remove variables with near zero variance
#names(train1)
Col.removed = nearZeroVar(train1)
#colnames(train1)[nearZeroVar(train1)]

# --- 1.3 Handling human error entries ---
# Replacing Garage_Yr_Blt = 2207 with 2007
train2 = train1
#train2 %>% ggplot(aes(Garage_Yr_Blt)) + geom_histogram()
#summary(train2$Garage_Yr_Blt)
train2$Garage_Yr_Blt[train2$Garage_Yr_Blt == 2207] = 2007
#summary(train2$Garage_Yr_Blt)

# --- 1.4 Removing correlated variables
train3=train2
#Gr_Liv_Area includes Second_Flr_SF, First_Flr_SF, so remove Gr_Liv_Area.

names.removed = c("PID", "Gr_Liv_Area", "Longitude", "Latitude")
Col.removed = append(Col.removed, which(names(train3) %in% names.removed))
#names(train2[,Col.removed])

train3 <- train3[, -Col.removed]

# --- 1.5 Applying winsorization
train4 = train3
winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_1", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF")

quan.value <- 0.95
for(var in winsor.vars){
  tmp <- train4[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  train4[, var] <- tmp
}

# --- 1.6 Apply Log transformation to Sale_Price ---
train5 = train4
train5$Sale_Price <- log(train5$Sale_Price)

train.x = train5[, colnames(train5)!="Sale_Price"]
train.y = train5$Sale_Price

# --- 1.7 Handle categorical features ---
categorical.vars <- colnames(train.x)[
  which(sapply(train.x,
               function(x) mode(x)=="character"))]
train.matrix <- train.x[, !colnames(train.x) %in% categorical.vars, 
                        drop=FALSE]
n.train <- nrow(train.matrix)
for(var in categorical.vars){
  mylevels <- sort(unique(train.x[, var]))
  m <- length(mylevels)
  m <- ifelse(m>2, m, 1)
  tmp.train <- matrix(0, n.train, m)
  col.names <- NULL
  for(j in 1:m){
    tmp.train[train.x[, var]==mylevels[j], j] <- 1
    col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
  }
  colnames(tmp.train) <- col.names
  train.matrix <- cbind(train.matrix, tmp.train)
}

####################
## 2. Train two models
####################

# --- 2.1 Lasso
# set.seed(0803)
# 
# model.lasso <- cv.glmnet(as.matrix(train.matrix), train.y, alpha = 1)
# sel.vars <- predict(model.lasso, type="nonzero", 
#                     s = model.lasso$lambda.1se)$X1
# model.lasso <- cv.glmnet(as.matrix(train.matrix[, sel.vars]), 
#                     train.y, alpha = 0)

# --- 2.2 Random Forest
set.seed(0803)
model.rf=randomForest(Sale_Price~., data=train5)

# --- 2.3 elastic net
set.seed(0803)
model.en <- cv.glmnet(as.matrix(train.matrix), train.y, alpha = 0.2)

####################
## 3. Preprocess test data
####################

test.org <- read.csv("test.csv")
## Preprocess test data, and save your prediction in two files

test1 = test.org

# --- 3.1 Handling missing values ---
# Replacing missing values on variable "Garage_Yr_Blt" with 0
test1[is.na(test1)] = 0

# --- 3.2 Remove variables with near zero variance

# --- 3.3 Handling human error entries ---
# Replacing Garage_Yr_Blt = 2207 with 2007
test2 = test1
test2$Garage_Yr_Blt[test2$Garage_Yr_Blt == 2207] = 2007

# --- 3.4 Removing correlated variables
test3=test2

#Gr_Liv_Area includes Second_Flr_SF, First_Flr_SF, so remove two variables.
#Total_Bsmt_SF = BsmtFin_SF_1 + BsmtFin_SF_2 + Bsmt_Unf_SF, so remove three variables.

# Remove variables
test3 <- test3[, -Col.removed]

# --- 3.5 Applying winsorization
test4 = test3
for(var in winsor.vars){
  tmp <- test4[, var]
  myquan <- quantile(train4[, var], probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  test4[, var] <- tmp
}

# --- 3.6 Apply Log transformation ---
test5 = test4

test.x = test5[, colnames(test5)!="Sale_Price"]
test.y = test5$Sale_Price

# --- 3.7 Handle categorical features ---
test.matrix <- test.x[, !colnames(test.x) %in% categorical.vars, 
                      drop=FALSE]

n.test <- nrow(test.matrix)

for(var in categorical.vars){
  mylevels <- sort(unique(train.x[, var]))
  m <- length(mylevels)
  m <- ifelse(m>2, m, 1)
  tmp.test <- matrix(0, n.test, m)
  col.names <- NULL
  for(j in 1:m){
    tmp.test[test.x[, var]==mylevels[j], j] <- 1
    col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
  }
  colnames(tmp.test) <- col.names
  test.matrix <- cbind(test.matrix, tmp.test)
}

####################
## 4. Prediction
####################

# --- 4.1 Lasso
# tmp.lasso <-predict(model.lasso, s = model.lasso$lambda.min, 
#               newx = as.matrix(test.matrix[, sel.vars]))
# tmp.lasso = exp(tmp.lasso) -1
# pred.lasso = data.frame(PID = test.org$PID, Sale_Price = tmp.lasso[,1])


# --- 4.2 Random Forest
tmp.rf = predict(model.rf, newdata = test5)
tmp.rf = exp(tmp.rf) - 1
pred.rf = data.frame(PID = test.org$PID, Sale_Price = tmp.rf)

# --- 4.3 elastic net
tmp.en <-predict(model.en, s = model.en$lambda.min, newx = as.matrix(test.matrix))
tmp.en = exp(tmp.en) -1
pred.en = data.frame(PID = test.org$PID, Sale_Price = tmp.en[,1])

####################
## 5. Output
####################
# write.table(pred.lasso, "mysubmission1.txt", sep=",", quote = FALSE)
write.table(pred.rf, "mysubmission2.txt", sep=",",quote = FALSE)
write.table(pred.en, "mysubmission1.txt", sep=",",quote = FALSE)

