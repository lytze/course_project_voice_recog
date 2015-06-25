

HMM and HDDA Based Human Voice Classification
========================================================
author: Li Yutze
date: 2015-06-25
css: custom.css
font-family: 'Calibri'

Introduction
========================================================
type: section

Dataset & Task

About the Dataset
========================================================

The 'JapaneseVowels' dataset:  
~~9 Speakers (as __classes__)~~  
~~The __same vowel ae__ is pronunced several times~~  
~~The raw records were sliced into short __pieces__, and for each piece~~  
~~a 12-degree Fourier prediction is performed to obtain the __spectrum__~~

```
                 F1       F2       F3      ...      F12  (Spectrum)
        T1     1.63     0.02     0.43      ...     0.14
        T2     1.55    -0.01     0.32      ...     0.12
       ...      ...      ...      ...      ...      ...
        Tn     1.39    -0.73     0.80      ...     0.10
  (Time series)
```

The dataset is constructed with 'blocks' of records  
~~12-columns represents 12 frequencies (__features__)~~  
~~And for each feature, follows a __various length time series__~~

Classification Task
========================================================

Each block is labeled by the recorder (9 speakers), and the task is:

~~__Given__: a block of record~~  
~~__Predict__: who is the speaker producing such record~~

Dataset sizes:

```
Training set:
     A      B      C      D      E      F      G      H      I 
    30     30     30     30     30     30     30     30     30
Testing set:
     A      B      C      D      E      F      G      H      I 
    31     35     88     44     29     24     40     50     29
```

~~Notice: the row-lengths of data blocks vary (7-24)~~

Results
========================================================
type: section

Hidden Markove Model based classification:  
~~Within sample accuracy: 99.6%~~  
~~Out-of sample accuracy: 98.4%~~

High-dimensional Discriminant Analysis based classification:  
~~Within sample accuracy: 98.9%~~  
~~Out-of sample accuracy: 98.4%~~ 

Time Serie Classification: HMM
========================================================
type: sub-section

Thought:  
~~Treat data as __time-series__ (as ther were), and use HMM to classify~~  
~~Well, HMMs are good at sequence analyses~~  

Method (briefly):  
~~Fit a __multivariete HMM__ for each class, so we get __9 HM models__~~  
~~Use the `forwardbackward` algorithm to calculate the __likelihood__ of~~  
~~given data blocks, and output the classes with __largest likelihoods__~~

Tools: R's Hidden Markov Packages
========================================================

Old `RHmm` Package:  
~~Not available in the current version of R (3.1)~~

Traditional `HMM` Package:  
~~Sutible for discrete time and discrete space HMMs~~

S3 based `HiddenMarkov` Package:  
~~Capable for liner and generalized-liner emission functions, single variable~~

S4 based `depmixS4` Package:  
~~Handles multivariate emissions with mixed models __(Used here)__~~

Data Formatting
========================================================

Read sizes (instance numbers) for each class and convert to numbers, then read the dataset as raw input lines




```r
> source('pipe.R') ## load my pipe operator
> classes <- LETTERS[1:9]
> train_size <- readLines('size_ae.train.txt') %>%
+     strsplit('\\ ') %>% `[[`(1) %>% as.numeric
> test_size <- readLines('size_ae.test.txt') %>%
+     strsplit('\\ ') %>% `[[`(1) %>% as.numeric
> train_lines <- readLines('ae.train.txt')
> test_lines <- readLines('ae.test.txt')
```

Data Formatting
========================================================

Format data into two (`training`/`testingg`) data frames, `*_lab` as class labels and `*_ins` marks instances


```r
> training <- read.table(text = train_lines)
> ends <- which(train_lines == '')
> starts <- c(1, ends[-length(ends)] + 1)
> training_cls <- rep(classes, times = train_size)
> training_ntm <- ends - starts
> training_ins <- sapply(classes, function(cls)
+     rep(1:train_size[classes == cls],
+         times = training_ntm[training_cls == cls])) %>% unlist
> training_lab <- rep(training_cls, training_ntm)
```

~~Testing set was processed similarly, not shown~~



Data Formatting
========================================================


```r
> case <- cbind(Class = training_lab, Instance = training_ins, training)
> rownames(case) <- NULL
> case[c(1:4, 21:24, 601:604), 1:7]
```

```
    Class Instance       V1        V2       V3        V4        V5
1       A        1 1.860936 -0.207383 0.261557 -0.214562 -0.171253
2       A        1 1.891651 -0.193249 0.235363 -0.249118 -0.112890
3       A        1 1.939205 -0.239664 0.258561 -0.291458 -0.041053
4       A        1 1.717517 -0.218572 0.217119 -0.228186 -0.018608
21      A        2 1.303905  0.067256 0.597720 -0.271474 -0.236808
22      A        2 1.288280  0.018672 0.631579 -0.355112 -0.119216
23      A        2 1.332021 -0.058744 0.601928 -0.347913 -0.053463
24      A        2 1.436550 -0.206565 0.641775 -0.391073 -0.103646
601     B        4 0.357779 -0.937038 0.183809 -0.487891  0.416678
602     B        4 0.399077 -0.996821 0.173008 -0.433203  0.478615
603     B        4 0.398805 -1.020785 0.161763 -0.392988  0.519761
604     B        4 0.312645 -1.006147 0.188503 -0.373804  0.575191
```

Train HMMs
========================================================


```r
> require(depmixS4)
> set.seed(1)
> fits <- sapply(classes, function(f) {
+     cls <- training[training_lab == f, ]
+     depmix(list(V1 ~ 1,  V2 ~ 1,  V3 ~ 1,  V4 ~ 1,
+                 V5 ~ 1,  V6 ~ 1,  V7 ~ 1,  V8 ~ 1,
+                 V9 ~ 1, V10 ~ 1, V11 ~ 1, V12 ~ 1),
+            data = cls,
+            family = list(gaussian(), gaussian(), gaussian(),
+                          gaussian(), gaussian(), gaussian(),
+                          gaussian(), gaussian(), gaussian(),
+                          gaussian(), gaussian(), gaussian()),
+            nstates = 5,
+            ntimes = training_ntm[training_cls == f]) %>%
+         fit(verbose = F)
+ })
```

~~Loop through all classes and fit for each a multivariate HMM,~~  
~~treat all __12 features as independent gaussian__ distributed variable~~  
~~and use __5 latend states__ to train the models.~~

Train HMMs
========================================================


```r
> summary(fits[[1]], which = 'transition')
```

```

Transition matrix 
           toS1      toS2      toS3     toS4      toS5
fromS1 9.58e-01 2.44e-274  4.22e-02 3.11e-71 4.08e-149
fromS2 1.72e-60  8.06e-01 1.52e-103 1.83e-01  1.16e-02
fromS3 2.90e-02 1.77e-286  9.71e-01 2.44e-79 1.31e-257
fromS4 7.56e-02 5.02e-110  1.60e-01 7.65e-01  1.85e-74
fromS5 4.19e-02  2.39e-72  9.76e-03 7.95e-02  8.69e-01
```

~~Transition matrix of fitted model for class A~~

Make Prediction
========================================================


```r
> predict_voice <- function(voice) {
+     init_mod <- depmix(
+         list(V1 ~ 1, V2 ~ 1, V3 ~ 1,  V4 ~ 1,  V5 ~ 1,  V6 ~ 1,
+              V7 ~ 1, V8 ~ 1, V9 ~ 1, V10 ~ 1, V11 ~ 1, V12 ~ 1),
+         data = voice,
+         family = list(
+             gaussian(), gaussian(), gaussian(), gaussian(),
+             gaussian(), gaussian(), gaussian(), gaussian(),
+             gaussian(), gaussian(), gaussian(), gaussian()),
+         nstates = 5)
+     ll <- sapply(1:9, function(i) {
+         mod <- setpars(init_mod, getpars(fits[[i]]))
+         forwardbackward(mod)$logLike
+     })
+     classes[which(ll == max(ll))[1]]
+ }
> c(predict_voice(testingg[12, ]),  testingg_lab[12],
+   predict_voice(testingg[405, ]), testingg_lab[405],
+   predict_voice(testingg[788, ]), testingg_lab[788])
```

```
[1] "A" "A" "I" "A" "B" "B"
```

Within Sample Accuracy
========================================================


```r
> inst <- paste(training_lab, training_ins)
> wi_pre_hmm <- sapply(inst %>% factor %>% levels, function(i) 
+     predict_voice(training[inst == i, ]))
> wi_conf_hmm <- caret::confusionMatrix(wi_pre_hmm, training_cls)
> print(wi_conf_hmm$table, zero.print = ''); wi_conf_hmm$overall[1]
```

```
          Reference
Prediction  A  B  C  D  E  F  G  H  I
         A 30                    1   
         B    30                     
         C       30                  
         D          30               
         E             30            
         F                30         
         G                   30      
         H                      29   
         I                         30
```

```
Accuracy 
   0.996 
```

Out of Sample Accuracy
========================================================


```r
> inst <- paste(testingg_lab, testingg_ins)
> ou_pre_hmm <- sapply(inst %>% factor %>% levels, function(i)
+     predict_voice(testingg[inst == i, ]))
> ou_conf_hmm <- caret::confusionMatrix(ou_pre_hmm, testingg_cls)
> print(ou_conf_hmm$table, zero.print = ''); ou_conf_hmm$overall[1]
```

```
          Reference
Prediction  A  B  C  D  E  F  G  H  I
         A 31                       1
         B    32                     
         C     2 88                  
         D          43               
         E             29           1
         F                24         
         G                   40      
         H     1                50   
         I           1             27
```

```
 Accuracy 
0.9837838 
```

Image Classification Based Method: HDDA
========================================================
type: sub-section

Thought:  
~~View data blocks as a gray-level images~~  
~~And use image classification methods to predict~~

What 'Voice-Images' Look Like
========================================================

If we treat __time__ and __feature__ as two dimensions, __Fourier coefficients__ as 'gray' levels, a single data block can be viewed like a picture

Let's test if there are some __fixed pattern__ hidden in the pictures

What 'Voice-Images' Look Like
========================================================

First we need a function to __resize__ data blocks into fixed sizes, since these blocks have various row numbers (time series length)  
~~HMM fitting does not care about the lengths, but image recognition cares~~


```r
> require(fields)
> compMat <- function(voice) {
+     interp.surface.grid(list(x = 1:nrow(voice), y = 1:12, z = voice),
+                         list(x = seq(1, nrow(voice), length = 12),
+                              y = 1:12))$z
+ }
```

Linear interpolation is applied to 'zoom' the matrix into 12x12 size

What 'Voice-Images' Look Like
========================================================

<img src="index-figure/unnamed-chunk-12-1.png" title="plot of chunk unnamed-chunk-12" alt="plot of chunk unnamed-chunk-12" style="display: block; margin: auto;" />

What 'Voice-Images' Look Like
========================================================

<img src="index-figure/unnamed-chunk-13-1.png" title="plot of chunk unnamed-chunk-13" alt="plot of chunk unnamed-chunk-13" style="display: block; margin: auto;" />

Reform Data
========================================================

Reform data into list of 12x12 matrices, and linearize them as a 144 dimension vector, then stack vectors into data frame.


```r
inst <- paste(training_lab, training_ins)
training <- sapply(inst %>% factor %>% levels, function(i) {
    compMat(training[inst == i, ]) %>% as.vector
}) %>% t %>% as.data.frame

inst <- paste(testingg_lab, testingg_ins)
testingg <- sapply(inst %>% factor %>% levels, function(i) { 
    compMat(testingg[inst == i, ]) %>% as.vector
}) %>% t %>% as.data.frame

training[1:4, 139:144] %>% `rownames<-`(NULL)
```

```
       V139       V140       V141        V142        V143      V144
1 0.1171748 0.12719936 0.02520936 -0.06525918 -0.14603182 -0.175986
2 0.1022945 0.04641364 0.02765800  0.03059891  0.02342836  0.034121
3 0.1637670 0.01564400 0.07353700  0.06521200  0.05742400  0.038621
4 0.2424693 0.15182945 0.09229282  0.07726864  0.08437555  0.023707
```

HDDA Works Well!
========================================================

__High-dimension discriminant analysis__ (HDDA) which is implemented in R's `HDclassif` package has been reported to have good performance in OCR (__optical character recognition__), faster and more accurate than even SVMs.

<center>![ocr](./index-figure/ocr.png)</center>

~~Examples of OCR input figures, pixel size = __16x16__~~  
~~similar to our __12x12__ problem!~~

Train HDDA
========================================================


```r
> require(HDclassif)
> set.seed(1)
> t_init <- Sys.time()
> fit_hdda <- hdda(data = training, cls = training_cls,
+                  model = 'all', show = F)
> t_lag <- Sys.time() - t_init
```

This model training is really fast:


```r
> t_lag
```

```
Time difference of 0.05163813 secs
```

Within Sample Accuracy
========================================================


```r
> wi_pre_hdda <- predict(fit_hdda, training)$class
> wi_conf_hdda <- caret::confusionMatrix(wi_pre_hdda, training_cls)
> print(wi_conf_hdda$table, zero.print = ''); wi_conf_hdda$overall[1]
```

```
          Reference
Prediction  A  B  C  D  E  F  G  H  I
         A 30                        
         B    29                     
         C     1 30           1      
         D          30               
         E             30            
         F                30         
         G                   29      
         H                      29   
         I                       1 30
```

```
 Accuracy 
0.9888889 
```

Out of Sample Accuracy
========================================================


```r
> ou_pre_hdda <- predict(fit_hdda, testingg)$class
> ou_conf_hdda <- caret::confusionMatrix(ou_pre_hdda, testingg_cls)
> print(ou_conf_hdda$table, zero.print = ''); ou_conf_hdda$overall[1]
```

```
          Reference
Prediction  A  B  C  D  E  F  G  H  I
         A 30                        
         B    34                     
         C       86              1   
         D          44        1      
         E             29            
         F                24         
         G                   39      
         H     1  1             49   
         I  1     1                29
```

```
 Accuracy 
0.9837838 
```

Summary
========================================================
type: section

<br>
<center>
<table class="shtable">
 <thead>
  <tr>
   <th style="text-align:center;"> Accuracy </th>
   <th style="text-align:center;"> HHM </th>
   <th style="text-align:center;"> HDDA </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:center;"> Within Sample </td>
   <td style="text-align:center;"> 99.6% </td>
   <td style="text-align:center;"> 98.9% </td>
  </tr>
  <tr>
   <td style="text-align:center;"> Out of Sample </td>
   <td style="text-align:center;"> 98.4% </td>
   <td style="text-align:center;"> 98.4% </td>
  </tr>
</tbody>
</table>
</center>

~~Within and out-of sample accuracies~~

Compare Models
========================================================

### HMM (as time series classification)

Really good accuracy  
~~Literature reported: 96.2%, Here: __98.4%__~~  
~~BTW: The data source literature model 'passing-through': 94.1%~~  
Time consuming  
~~Fitting model for about __20s__ and predicting (single case) for __0.23s__~~

### HDDA (as image recognition)

Fast fitting and fast predicting  
~~Training model in __0.1 sec__, predicting all in __0.1 sec__~~  
~~But: __reshaping__ data matrices requires time (__7 sec__ for all)~~

Reference
========================================================

Literatures / Package Vignettes

<small>"Multidimensional curve classifcation using passing-through regions", __M Kudo et al__, _Pattern Recognition Letters (20)_</small>  
~~Data Source~~

<small>"depmixS4: An R Package for Hidden Markov Models", __I Visser & M Speekenbrink__, _Journal of Statistical Software (36)_</small>  
~~HMM Package __depmixS4__~~

<small>"HDclassif: An R Package for Model-Based Clustering and Discriminant Analysis of High-Dimensional Data", __L Berge, C Bouveyron & S Girard__, _Journal of Statistical Software (46)_</small>  
~~HDDA Package __HDclissif__~~

Reference
========================================================

Helpful Resources

<small>"Sequence Classification: with emphasis on Hidden Markov Models and Sequence Kernels", __A M White__, [cs.unc.edu/~lazebnik](www.cs.unc.edu/~lazebnik/fall09/sequence_classification.pdf)</small>  
~~A good __slide show on HMMs__, clearly explains what is HMM and the~~  
~~logics of HMMs' 3 problem: evaluating, decoding & learning~~

<small>"Hidden Markov Models – Model Description / Forward & Viterbi Algorithm / Examples In R / Trend Following", __GekkoQuant__, [gekkoquant.com (blog)](gekkoquant.com/2014/05/18/hidden-markov-models-model-description-part-1-of-4/)</small>  
~~A demo of RHmm package, explains __2 ways of classification using HMMs__~~  
~~the though is helpful, even though RHmm is an expired~~
