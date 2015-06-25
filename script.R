## Load HiddenMarkov package
require(depmixS4)

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

## HMM method
## ## Fit 9 HMMs
set.seed(1)
fits <- sapply(classes, function(f) {
    cls <- training[training_lab == f, ]
    depmix(list(V1 ~ 1, V2 ~ 1, V3 ~ 1, V4 ~ 1, V5 ~ 1, V6 ~ 1,
                V7 ~ 1, V8 ~ 1, V9 ~ 1, V10 ~ 1, V11 ~ 1, V12 ~ 1),
           data = cls,
           family = list(gaussian(), gaussian(), gaussian(), gaussian(),
                         gaussian(), gaussian(), gaussian(), gaussian(),
                         gaussian(), gaussian(), gaussian(), gaussian()),
           nstates = 5, ntimes = training_ntm[training_cls == f]) %>%
        fit(verbose = F)
})
## ## Make the prediction function
predict_voice <- function(voice) {
    ## INFO
    ## ## Predict the class of `voice`
    ## ARGS
    ## ## voice: data frame of 12 colums
    ## RETN
    ## ## class label (charactar)
    init_mod <- depmix(list(V1 ~ 1, V2 ~ 1, V3 ~ 1, V4 ~ 1, V5 ~ 1, V6 ~ 1,
                            V7 ~ 1, V8 ~ 1, V9 ~ 1, V10 ~ 1, V11 ~ 1, V12 ~ 1),
                       data = voice,
                       family = list(gaussian(), gaussian(), gaussian(), gaussian(),
                                     gaussian(), gaussian(), gaussian(), gaussian(),
                                     gaussian(), gaussian(), gaussian(), gaussian()),
                       nstates = 5)
    ll <- sapply(1:9, function(i) {
        mod <- setpars(init_mod, getpars(fits[[i]]))
        forwardbackward(mod)$logLike
    })
    classes[which(ll == max(ll))[1]]
}
## ## Calculate within-sample error rate
inst <- paste(training_lab, training_ins)
wi_pre <- sapply(inst %>% factor %>% levels, function(i) {
    predict_voice(training[inst == i, ])
})
wi_conf <- caret::confusionMatrix(wi_pre, training_cls)
wi_conf
## ## Calculate out-of-sample error rate
inst <- paste(testingg_lab, testingg_ins)
ou_pre <- sapply(inst %>% factor %>% levels, function(i) {
    predict_voice(testingg[inst == i, ])
})
ou_conf <- caret::confusionMatrix(ou_pre, testingg_cls)
ou_conf

## ANN method
lis <- paste(rep(LETTERS[1:4], each = 3), rep(1:3, 4))
layout(matrix(1:12, 3, 4))
par(mar = c(0.1, 0.1, 0.1, 0.1), oma = c(0, 0, 3, 0))
n <- 1
for (i in lis) {
    if (n %% 3 == 0)
        axis(side = 3, outer = T, labels = LETTERS[n / 3], at = 0.5)
    n <- n + 1
    mat <- testingg[inst == i, ] %>% as.matrix
    mat %>%
        image(frame = T, axes = F, col = gray.colors(2))
}
n <- 1
for (i in lis) {
    if (n %% 3 == 0)
        axis(side = 3, outer = T, labels = LETTERS[n / 3], at = 0.5)
    n <- n + 1
    mat <- testingg[inst == i, ] %>% as.matrix
    mat %>% `>`(mean(mat)) %>%
        image(frame = T, axes = F, col = gray.colors(2))
}
n <- 1
for (i in lis) {
    if (n %% 3 == 0)
        axis(side = 3, outer = T, labels = LETTERS[n / 3], at = 0.5)
    n <- n + 1
    mat <- testingg[inst == i, ] %>% as.matrix
    mat %>% apply(1, function(col) col > mean(col)) %>% t %>%
        image(frame = T, axes = F, col = gray.colors(2))
}
require(fields)
compMat <- function(voice) {
    ## INFO
    ## ## Compress `voice` record into 12 x 12 matrix 
    interp.surface.grid(list(x = 1:nrow(voice), y = 1:12, z = voice),
                        list(x = seq(1, nrow(voice), length = 12),
                             y = 1:12))$z
}