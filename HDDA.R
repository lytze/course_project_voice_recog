## Load caret package
require(HDclassif)

## Shift to working dir
setwd('~/Documents/Courses/zju_bioinformatics_III/data_mining_and_machine_learning/play_jap_vowel/')

## Prepare piping operator
source('pipe.R')

## Load data
classes <- LETTERS[1:9]
train_size <- readLines('size_ae.train.txt') %>% strsplit('\\ ') %>% `[[`(1) %>% as.numeric
test_size <- readLines('size_ae.test.txt') %>% strsplit('\\ ') %>% `[[`(1) %>% as.numeric
train_lines <- readLines('ae.train.txt')
test_lines <- readLines('ae.test.txt')

training <- read.table(text = train_lines)
ends <- which(train_lines == '')
starts <- c(1, ends[-length(ends)] + 1)
training_cls <- rep(classes, times = train_size)
training_ntm <- ends - starts
training_ins <- sapply(classes, function(cls)
    rep(1:train_size[classes == cls],
        times = training_ntm[training_cls == cls])) %>% unlist
training_lab <- rep(training_cls, training_ntm)

testingg <- read.table(text = test_lines)
ends <- which(test_lines == '')
starts <- c(1, ends[-length(ends)] + 1)
testingg_cls <- rep(classes, times = test_size)
testingg_ntm <- ends - starts
testingg_ins <- sapply(classes, function(cls)
    rep(1:test_size[classes == cls],
        times = testingg_ntm[testingg_cls == cls])) %>% unlist
testingg_lab <- rep(testingg_cls, testingg_ntm)

rm(train_lines, train_size, test_lines, test_size)

## ANN method
require(fields)
compMat <- function(voice) {
    ## INFO
    ## ## Compress `voice` record into 12 x 12 matrix 
    interp.surface.grid(list(x = 1:nrow(voice), y = 1:12, z = voice),
                        list(x = seq(1, nrow(voice), length = 12),
                             y = 1:12))$z
}
inst <- paste(training_lab, training_ins)
lis <- paste(rep(LETTERS[1:4], each = 3), rep(1:3, 4))
layout(matrix(1:12, 3, 4))
par(mar = c(0.1, 0.1, 0.1, 0.1), oma = c(0, 0, 3, 0))
n <- 1
for (i in lis) {
    if (n %% 3 == 0)
        axis(side = 3, outer = T, labels = LETTERS[n / 3], at = 0.5)
    n <- n + 1
    mat <- training[inst == i, ] %>% as.matrix %>% compMat %>%
        image(frame = T, axes = F, col = gray.colors(12))
}

## ## Reshape all instance
inst <- paste(training_lab, training_ins)
training <- sapply(inst %>% factor %>% levels, function(i) {
    compMat(training[inst == i, ]) %>% as.vector
}) %>% t %>% as.data.frame
inst <- paste(testingg_lab, testingg_ins)
testingg <- sapply(inst %>% factor %>% levels, function(i) { 
    compMat(testingg[inst == i, ]) %>% as.vector
}) %>% t %>% as.data.frame

set.seed(1)

## HDDA model
fit_hdda <- hdda(data = training, cls = training_cls, model = 'all')
pre_wi <- predict(fit_hdda, training)$class
caret::confusionMatrix(pre_wi, training_cls)
pre_ou <- predict(fit_hdda, testingg)$class
caret::confusionMatrix(pre_ou, testingg_cls)

## Radial Basis Function Kernel
require(caret)
training <- data.frame(y = training_cls, training)
testingg <- data.frame(y = testingg_cls, testingg)
fit_svm <- train(y ~ ., data = training, method = 'svmRadial', sigma = 0.1, C = 10)
