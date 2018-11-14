Predicting Singapore Open Masters Scores
================
Hui Chiang

Introduction
------------

In this project, I want to look at whether we can predict a bowler's Masters Final score based on their MQ scores, and other factors such as sex and category. The data set we will be looking at specifically is the 2012 Singapore Open (because it was the last one I participated in heh), and the U18, Youth (U21) and Open categories for both sexes.

Extracting the Data
-------------------

Let's first take a look at the ABF website (2012) where I obtained the scores from.

<img src="/Users/tayhuichiang94/Desktop/personal%20project/SG_Open_2012/u18.png" alt="Fig 1.1: U18 MQ Scores" width="300" /> <img src="/Users/tayhuichiang94/Desktop/personal%20project/SG_Open_2012/u18masters.png" alt="Fig 1.2: U18 Masters Scores" width="300" />

We want to extract the names, country, series 1 and 2 (mq1, mq2) and high game of the 16 qualifiers in Figure 1.1, after which we can easily obtain the average. We also want to extract the finals scores of the 16 qualifiers in Figure 1.2.

We do this by using regular expressions (regex) to scrape the required information from the html source code of the webpage. Regex is a way for us to search the large html file for certain strings that match our desired pattern (Goyvaerts, 2018).

After extraction, we have a dataframe for the MQ scores,

    ## Loading required package: Matrix

    ## Loading required package: foreach

    ## Loaded glmnet 2.0-13

    ##                    names country mq1 mq2  hg  average
    ## 1 ALEXANDER TAN KEE HOCK     SIN 735 590 285 220.8333
    ## 2           MARCUS LEONG     SIN 658 626 276 214.0000
    ## 3   MUHD DANIEL ZHENG ZI     SIN 640 630 234 211.6667
    ## 4         TAY HUI CHIANG     SIN 648 622 231 211.6667
    ## 5             JAVIER TAN     SIN 659 606 235 210.8333
    ## 6   NICHOLAS GOH SHI HAO     SIN 652 606 252 209.6667

and one for the Masters scores,

    ##                    names  score
    ## 1             JAVIER TAN 217.50
    ## 2         TAY HUI CHIANG 203.38
    ## 3           NG KAI XIANG 201.13
    ## 4     BASIL NG JUI CHANG 197.88
    ## 5 ALEXANDER TAN KEE HOCK 195.13
    ## 6         ANDRIY HURWOOD 191.00

We now wish to perform an inner join to merge the 2 dataframes on the names of the bowlers, such that we can obtain a single dataframe which contains both te MQ and the Masters scores for each bowler.

``` r
MU182012 <- merge(MU182012MQ, MU182012Masters, 'names')
dim(MU182012)
```

    ## [1] 15  7

When we check the dimensions of the dataframe, we see that there are only 15 bowlers, instead of 16. This could be because one name was keyed in differently on the different webpages. Checking for differences in names,

    ## [1] "MUHD DANIEL ZHENG ZI"

    ## [1] "MUHD DANIAL ZHENG YI"

Indeed, we find that there is a name that has been keyed in differently. We correct this and merge the dataframes again.

Finally, since we will also be looking at scores from other categories and sexes, we add the 'category' and 'sex' columns. Our final dataframe for the Boys U18 division looks like this,

    ##                    names country mq1 mq2  hg  average  score category  sex
    ## 1 ALEXANDER TAN KEE HOCK     SIN 735 590 285 220.8333 195.13      U18 Male
    ## 2    ANDERS KHOO TENG YI     SIN 624 612 236 206.0000 180.88      U18 Male
    ## 3         ANDRIY HURWOOD     SLO 647 606 240 208.8333 191.00      U18 Male
    ## 4     BASIL NG JUI CHANG     SIN 629 579 215 201.3333 197.88      U18 Male
    ## 5  BENEDICT TAN XUAN WEI     SIN 648 605 235 208.8333 187.75      U18 Male
    ## 6     BRYAN LEE OON SENG     SIN 605 602 229 201.1667 179.00      U18 Male

We extract the data for the other categories in the same way. We then do a left join on each dataframe to create an overall dataframe for the 6 different categories (3 categories for 2 sexes).

Also do note that for the Open category, we do not consider the Desperado and Defending Champion because they may not have a corresponding MQ score, and that we will use their Round 1 Finals score for analysis.

Taking a look at a summary of our dataframe,

    ##     names              country        mq1             mq2       
    ##  Length:127         SIN    :72   Min.   :585.0   Min.   :552.0  
    ##  Class :character   MAS    :27   1st Qu.:638.5   1st Qu.:601.5  
    ##  Mode  :character   KOR    :13   Median :661.0   Median :623.0  
    ##                     TPE    : 4   Mean   :666.9   Mean   :623.4  
    ##                     INA    : 3   3rd Qu.:695.0   3rd Qu.:643.5  
    ##                     PHI    : 3   Max.   :787.0   Max.   :717.0  
    ##                     (Other): 5                                  
    ##        hg         average          score       category      sex    
    ##  Min.   :213   Min.   :194.0   Min.   :163.2   Open:67   Female:49  
    ##  1st Qu.:235   1st Qu.:207.4   1st Qu.:184.5   U18 :24   Male  :78  
    ##  Median :247   Median :213.3   Median :192.6   U21 :36              
    ##  Mean   :249   Mean   :215.0   Mean   :193.9                        
    ##  3rd Qu.:259   3rd Qu.:223.1   3rd Qu.:202.0                        
    ##  Max.   :298   Max.   :239.5   Max.   :228.3                        
    ## 

We see that the most bowlers are from 3 countries, Singapore, Malaysia and Korea. We will group the other countries into 'Others' to avoid possible problems with cross-validation.

Exploratory Data Analysis
-------------------------

Let's look at some possible trends within the data. ![](SG_Open_2012_files/figure-markdown_github/unnamed-chunk-10-1.png)

We see that our continuous variables, mq1, mq2, and hg all seem to be positively correlated with each other. We do expect a bowler's high game to have a strong positive correlation with the mq scores.

![](SG_Open_2012_files/figure-markdown_github/unnamed-chunk-11-1.png)

From the above two figures, we see that while average is positively correlated with finals score, this could be due to the effect of the differences in score arising from the different categories and sexes.

Lastly, we would like to take a look at the performance of each country in the Open category, ![](SG_Open_2012_files/figure-markdown_github/unnamed-chunk-12-1.png)

In the Masters Final, Korean bowlers bowled better on average than all the other countries, although Singaporean bowlers exhibited the largest variance. As for the MQ scores, Korean MQs were only slightly higher on average than the others, while Singaporean MQ scores were the lowest, possibly due to the presence of a 'Local Pool'.

Linear Regression
-----------------

Linear regression is an approach used to model the relatioship between a continuous variable (in our case Masters Score) with other independent, predictor variables (mq scores, country etc). Intuititvely, we are trying to find a line of best-fit, such as the one in the figure below. ![](SG_Open_2012_files/figure-markdown_github/unnamed-chunk-13-1.png)

Linear regression assumes a linear relationship between the variables. Say we have *n* data points and *p* predictors, then we form the relationship *Y*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>*i*1</sub> + … + *β*<sub>*p*</sub>*X*<sub>*i**p*</sub> + *ϵ* + *i* where *i* = 1…*n*. In this equation, the *β* represent the coefficients which we want to estimate, and *ϵ* is a random error term.

We use a matrix representation for our data, **Y** = **X****β** + **ϵ** where $\\mathbf{Y} =\\begin{pmatrix}Y\_{1}\\\\ \\vdots \\\\ Y\_{n}\\end{pmatrix}$, $\\mathbf{X}=\\begin{pmatrix}1 & X\_{11} & \\dots & X\_{1p} \\\\ \\vdots \\\\ 1 & X\_{n1} & \\dots & X\_{np}\\end{pmatrix}$, $\\boldsymbol{\\beta} = \\begin{pmatrix}\\beta\_{1}\\\\ \\vdots \\\\ \\beta\_{p} \\end{pmatrix}$ and $\\boldsymbol{\\epsilon} = \\begin{pmatrix}\\epsilon\_{1}\\\\ \\vdots \\\\ \\epsilon\_{n} \\end{pmatrix}$.

We estimate **β** by minimising mean squared error, $MSE=\\frac{1}{n}\\left(\\mathbf{Y}-\\mathbf{X}\\boldsymbol{\\widehat{\\beta}}\\right)^{T}\\left(\\mathbf{Y}-\\mathbf{X}\\boldsymbol{\\widehat{\\beta}}\\right)$, where $\\boldsymbol{\\widehat{\\beta}}$ is a quantity we estimate. This gives us the solution $\\boldsymbol{\\widehat{\\beta}}=\\left(\\mathbf{X^{T}X}\\right)^{-1}\\mathbf{X}^{T}\\mathbf{Y}$. We can then use $\\boldsymbol{\\widehat{\\beta}}$ to predict the Masters scores given the predictor variables.

We will test our model using cross-validation. The idea behind cross-validation is that we first randomly divide our data into *k* different groups, each with approximately the same number of entries. We leave out one group as the test set, and use the rest of the entries in the other *k* − 1 groups to train our model. Then, we test the performance of the model on the test set to obtain the mean squared error for that set. We repeat this process *k* times to find the cross-validation error, $\\frac{1}{k}\\sum\_{i=1}^{k}MSE\_{i}$.

Now, let's try it out!

``` r
#We use 10 fold cross-validation here, which means we set k=10
#Creating the linear model
mylm <- glm(score~mq1+mq2+hg+country+sex+category, data=Combined2012)
mycv <- cv.glm(Combined2012, mylm, K=10)
mycv$delta[1]
```

    ## [1] 147.2356

We obtain a mean squared error of around 147, that is huge! It does not bode well in our attempt to predict Masters scores.

Ridge Regression
----------------

Ridge regression is similar to linear regression, except that we introduce an additional term to penalise the coefficients if they get too large. In matrix form, we now wish to minimise $\\left(\\mathbf{Y}-\\mathbf{X}\\boldsymbol{\\widehat{\\beta}}\\right)^{T}\\left(\\mathbf{Y}-\\mathbf{X}\\boldsymbol{\\widehat{\\beta}}\\right)+\\lambda\\boldsymbol{\\widehat{\\beta}}^{T}\\boldsymbol{\\widehat{\\beta}}$, where *λ* is a constant of our choice. At *λ* = 0, this reduces to our original least squares regression. We can form our estimate, $\\boldsymbol{\\widehat{\\beta}}=\\left(\\mathbf{X}^{T}\\mathbf{X}+\\lambda\\mathbf{I}\\right)^{-1}$.

Due to the introduction of the diagonal term, *λ***I**, ridge regression is more table than ordinary least squares regression, especially with multicollinear data (Hastie, Tibshirani & Friedman, 2001), which could be applicable here. We also note that there is a need to standardise the data due to the additional penalising term. We will make use of R's cv.glmnet function, which is in the glmnet library.

``` r
#cv.glmnet requires us to convert our dataframe into a matrix
X <- model.matrix(~0+country+mq1+mq2+hg+country+sex+category, data=Combined2012)
Y <- data.matrix(Combined2012$score)

#Alpha represents the way we penalise the coefficents, alpha=0 is what we require, alpha=1
#penalises using the absolute value of the coefficients.
myrr <- cv.glmnet(x=X, y=Y, alpha=0, standardize=TRUE, intercept=TRUE)
```

We can now look at the results of our ridge regression. We have the minimum mean cross-validation error out of all the lambda that was tried,

``` r
min(myrr$cvm)
```

    ## [1] 139.4587

and we have the lambda that gave us this minimum,

``` r
myrr$lambda.min
```

    ## [1] 8.790355

Unfortunately, we see that ridge regression did not improve our prediction by much at all!

K-Nearest Neighbours
--------------------

Lastly, we will be using the K-Nearest Neighbours (KNN) algorithm. KNN is a non-parametric method more frequently used as a classifier, but can function for regression as well. In our case, when we wish to predict the finals score of a bowler, we first measure the 'distance' between the predictor variables of the point we wish to predict, and all other points. Then, we find the scores of the K closest neighbours and use the mean as our prediction. Again, we will need to standardise the variables in order to prevent a single distance from dominating the others. We will use Euclidean distance for the continuous variables, and Hamming distance for the categorical vavriables.

We first standardise our data,

    ##   country        mq1         mq2         hg category  sex  score
    ## 1     MAS  0.3389228  1.18168183  0.4058175     Open Male 199.60
    ## 2     MAS  1.7565132  0.08051291  0.7528803     Open Male 197.30
    ## 3     MAS -1.2590881 -0.23410678 -1.5608717      U21 Male 220.00
    ## 4     SIN -0.4085339 -1.17796585 -0.6353709      U21 Male 177.38
    ## 5     MAS  0.9832821  1.71653530  1.5626936     Open Male 195.20
    ## 6     SIN -0.6405032  0.48951851 -0.8667461      U21 Male 176.63

Now, we write a function to perform KNN regression for us,

``` r
#We require a training dataset, a point to predict, and a value for K
KNN.regress <- function (train, new, K){
  #Calculate the difference between the continuous variables 
  new_df <- data.frame('mq1' = new$mq1 - train$mq1,
                       'mq2' = new$mq2 - train$mq2,
                       'hg' = new$hg - train$hg)
  dists <- sqrt(rowSums(new_df^2))
  
  #Add the difference between the categorical variables
  dists <- dists + 
            as.numeric(new$category != train$category) + 
            as.numeric(new$sex != train$sex) +
            as.numeric(new$country != train$country)
  
  #Sort the points and find the K closest points
  neighbours  <- order(dists)[1:K]
  #Return a prediction
  pred <- mean(train$score[neighbours])
  return(pred)
}
```

We split our dataset for cross-validation using a solution provided by Drew (2014) and run our KNN.regress function. We will run the function for *K* = 1, …, 10 and see which gives us the lowest cross-validation error. ![](SG_Open_2012_files/figure-markdown_github/unnamed-chunk-20-1.png)

We can see from our plot that from around *K* = 6 onwards, the cross-validation error reaches around 140 and does not change by much. Again, it does not seem that we can predict Masters scores with KNN regression.

Conclusion and Further Improvement
----------------------------------

While writing the code for this project, one way which I will look to improve on is trying to somehow automate the process through which I scraped the data from the ABF website, as doing it individually for each category was time-consuming and inefficient. I could have also provided some simple summary statistics and conduct hypothesis testing during the Exploratory Data Analysis to show that there is little to no correlation, but where's the fun in that!

We note that the performance of all 3 methods seem to suggest that we cannot accurately predict a bowler's Masters scores from his/her MQ scores. This is actually in line with what most bowlers believe, that anybody has an equal chance of winning regardless of how high they qualified with.

More importantly, we should recognise that even if we could predict the scores, the way this small project was conducted is invalid. This is because all the Masters scores appear on the same day, which means that we will obtain the training data at the same time as the data for those that we wish to predict! The reason I did this project then, was to satisfy some internal curiosity within me whether it was possible to predict the scores at all (it isn't).

A more realistic study could be to look at the qualifying and winning averages over the years, and using that try to predict the scores of the next year's tournament!

References
----------

1.  Asian Bowling Federation. (2012, May). *45th Singapore International Open Youth Boys U-18 Qualifying*. Retrieved from <http://www.abf-online.org/results/2012/45thspore-1.asp?Division=BU18>.

2.  Goyvaerts, J. (2018, October). *Regular Expressions Quick Start*. Retrieved from <https://www.regular-expressions.info/quickstart.html>

3.  Hastie, T., Tibshirani, R., Friedman, J. (2001). *The Elements of Statistical Learning*. New York, NY, USA: Springer New York Inc.

4.  Jake Drew. (2014, July). *How to split a data set to do 10-fold cross validation*. Retrieved from <https://stats.stackexchange.com/q/105839>.
