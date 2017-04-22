train <- read.csv("~/Desktop/DSG17/data/train.csv")

train$genre_id <- as.factor(train$genre_id)
# train$ts_listen <- as.factor(train$ts_listen)
train$media_id <- as.factor(train$media_id)
train$album_id <- as.factor(train$album_id)
train$context_type <- as.factor(train$context_type)
# train$release_date <- as.Date(train$release_date, format="%Y%m%d", origin="20000101")
train$platform_name <- as.factor(train$platform_name)
train$platform_family <- as.factor(train$platform_family)

train$listen_type <- as.factor(train$listen_type)
train$user_gender <- as.factor(train$user_gender)
train$user_id <- as.factor(train$user_id)
train$artist_id <- as.factor(train$artist_id)
train$is_listened <- as.factor(train$is_listened)

dropc <- c("release_date")
train <- train[,!names(train) %in% dropc]

########### Test ##########
test <- read.csv("~/Desktop/DSG17/data/test.csv")

test$genre_id <- as.factor(test$genre_id)
# test$ts_listen <- as.factor(test$ts_listen)
test$media_id <- as.factor(test$media_id)
test$album_id <- as.factor(test$album_id)
test$context_type <- as.factor(test$context_type)
# test$release_date <- as.Date(test$release_date, format="%Y%m%d", origin="20000101")
test$platform_name <- as.factor(test$platform_name)
test$platform_family <- as.factor(test$platform_family)

test$listen_type <- as.factor(test$listen_type)
test$user_gender <- as.factor(test$user_gender)
test$user_id <- as.factor(test$user_id)
test$artist_id <- as.factor(test$artist_id)
dropc <- c("sample_id", "release_date")
test <- test[,!names(test) %in% dropc]

################ Manual Reduction of categorical features ################################
n = 3000
train$genre_id <- as.numeric(train$genre_id)
ugid <- subset(table(as.factor(train$genre_id)), table(as.factor(train$genre_id)) > n)
train$genre_id[!(as.factor(train$genre_id) %in% as.data.frame(ugid)$Var1)] <- 99999999
train$genre_id <- as.factor(train$genre_id)
# summary(train$genre_id)

train$media_id <- as.numeric(train$media_id)
n = quantile(as.data.frame(table(train$media_id))$Freq, 0.995)
umid <- subset(table(as.factor(train$media_id)), table(as.factor(train$media_id)) > n)
train$media_id[!(as.factor(train$media_id) %in% as.data.frame(umid)$Var1)] <- 99999999
train$media_id <- as.factor(train$media_id)
# summary(train$media_id)

train$album_id <- as.numeric(train$album_id)
n = quantile(as.data.frame(table(train$album_id))$Freq, 0.98)
uaid <- subset(table(as.factor(train$album_id)), table(as.factor(train$album_id)) > n)
train$album_id[!(as.factor(train$album_id) %in% as.data.frame(uaid)$Var1)] <- 99999999
train$album_id <- as.factor(train$album_id)
# summary(train$album_id)

train$context_type <- as.numeric(train$context_type)
n = 1000
ucid <- subset(table(as.factor(train$context_type)), table(as.factor(train$context_type)) > n)
train$context_type[!(as.factor(train$context_type) %in% as.data.frame(ucid)$Var1)] <- 99999999
train$context_type <- as.factor(train$context_type)
# summary(train$context_type)

lkp <- as.data.frame(table(train$user_id))
lkp$FreqLog <- as.integer(log(lkp$Freq))
train$user_id <- lkp$FreqLog[match(train$user_id, lkp$Var1)]
train$user_id <- as.factor(train$user_id)

train$artist_id <- as.numeric(train$artist_id)
n = quantile(as.data.frame(table(train$artist_id))$Freq, 0.95)
uarid <- subset(table(as.factor(train$artist_id)), table(as.factor(train$artist_id)) > n)
train$artist_id[!(as.factor(train$artist_id) %in% as.data.frame(uarid)$Var1)] <- 99999999
train$artist_id <- as.factor(train$artist_id)
# summary(train$artist_id)

################ Repeat for Test: categorical features ################################
n = 3000
test$genre_id <- as.numeric(test$genre_id)
test$genre_id[!(as.factor(test$genre_id) %in% as.data.frame(ugid)$Var1)] <- 99999999
test$genre_id <- as.factor(test$genre_id)

test$media_id <- as.numeric(test$media_id)
test$media_id[!(as.factor(test$media_id) %in% as.data.frame(umid)$Var1)] <- 99999999
test$media_id <- as.factor(test$media_id)

test$album_id <- as.numeric(test$album_id)
test$album_id[!(as.factor(test$album_id) %in% as.data.frame(uaid)$Var1)] <- 99999999
test$album_id <- as.factor(test$album_id)

test$context_type <- as.numeric(test$context_type)
test$context_type[!(as.factor(test$context_type) %in% as.data.frame(ucid)$Var1)] <- 99999999
test$context_type <- as.factor(test$context_type)

lkp <- as.data.frame(table(test$user_id))
lkp$FreqLog <- as.integer(log(lkp$Freq))
test$user_id <- lkp$FreqLog[match(test$user_id, lkp$Var1)]
test$user_id <- as.factor(test$user_id)

test$artist_id <- as.numeric(test$artist_id)
test$artist_id[!(as.factor(test$artist_id) %in% as.data.frame(uarid)$Var1)] <- 99999999
test$artist_id <- as.factor(test$artist_id)


# n = 3000
# re = seq(1000, 40000, by=1000)
# nf = seq(1000, 40000, by=1000)
# i=0
# for (n in seq(1000, 20000, by=1000)){
#   ugid <- subset(table(train$genre_id), table(train$genre_id) > n)
#   nf[i] = length(ugid)
#   re[i] = length(subset(train$genre_id, train$genre_id %in% as.data.frame(ugid)$Var1))
#   i = i+1
# }


############## Preprocessing ####################
library(vtreat)
treatmentsC = designTreatmentsC(train,colnames(train),'is_listened', outcometarget = 1)
trainCT <- prepare(treatmentsC,train,pruneSig=1.0,scale=TRUE)
testCT <- prepare(treatmentsC,test,pruneSig=1.0,scale=TRUE)

