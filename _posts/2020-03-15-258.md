---
layout: post
title:  "Recommender System & Rating Prediction of Amazon Digital Music"
date:   2020-03-15
excerpt: "A couple of models are built based on the similarities among items (Jaccard similarity-based model), the significant features of customers and items (latent factor models), and the laws in review texts (text-analysis based model)."
project: true
tag:
- jekyll 
- moon
- blog
- about
- theme
comments: false
---


<font color="darkgrey"> Contents </font>
List Of Contents
Introduction
Data Exploratory Analysis
2.1 Dataset Overview
2.2 Data Cleaning
2.3 Exploratory Results
Literature review
Methodology
4.1. Model Evaluation
4.2. Baseline Model 
4.3 Models
4.3.1 Jaccard similarity-based model
4.3.2 Latent factor model
4.3.3 Text-analysis based model 
4.3.4 Other Models
Model analysis 
Results and Conclusion
References


<font color="darkgrey"> Introduction </font>
Since the onset of the digital revolution, consumers want to search for reviews before purchasing any product. In order to identify the sentiment based off of the reviews, various machine learning algorithms come into picture to help give an efficient rating on the basis of reviews which users can depend upon. In our report, we aim to build such models and come out with the most efficient one. Models like TF-IDF which targets text-analyzing and Latent-Factor which uses latent feature spaces to represent preferences of customers and features of items come to our rescue when it comes to predicting the ratings from reviews.


<font color="darkgrey"> Data Exploratory Analysis </font>
2.1 Dataset Overview
This Amazon Digital Music dataset (74.7MB) originally has 169781 records, with some values of certain features missing. It includes features as following:
● asin:​ item ID
● overall​: ratings
● reviewText:​ detailed review content
● reviewerID:​ reviewer’s unique ID,
● reviewName:​ the name of the reviewer
● style​: style of the purchased product
● summary​: summary of reviews
● vote:​ number of votes
Considering that there are plenty of missing values and the dataset is large enough to conduct our analysis, data preprocessing is done in the first place.
2.2 Data Cleaning
Before we load data into the workplace, we changed the expression of boolean values by assigning “True” to “true” and “False” to “false”. After we load the data, we fill the null values in features, such as [image] and [vote] in the data, by filling them with 0. We also changed the type of [review time] from string to datetime in order to do some exploratory analysis such as the distribution of the time of each record.
With selected entries without null values, we set the seed of 1234 and randomly choose 60,000 rows of data as our database because of the limited operating capacity of our computers. Then, we split our dataset to train, valid and test set with a proportion of 4:1:1.
2.3 Exploratory Results
The final dataset used in this project contains 60,000 entries, including the transaction and review data of 15549 unique customers and 10463 unique items, covering a time period from 1998/8/21 to 2018/9/26, and the distribution of the data during this period is shown by Figure 1.
According to the time distribution, we can see that the majority of the data lies in years after 2012. This may be reasonable because this may be the point online shopping platforms are rapidly developing and becoming increasingly mature that people started gaining trust in the online shopping portals. We also compared average ratings across different time periods and found that there is no obvious trend so we put aside time when considering our predictors.
Products in the dataset have an average rating of 4.707, which is very high. Given that users can only give rating in integers, most of the users must be giving 5 stars. The following figure shows the distribution of reviews in terms of ratings. Most of the ratings are concentrated on 5, which is in line with a low standard deviation of 0.705. Since the variation of the rating levels are low, even the MSE of a baseline model may be very low and may be hard to beat. This brings challenge to our predictions.
As for verified and unverified reviews, the average rating is slightly different. There are 7061 unverified users who constitute an average rating of 4.428. Whereas, there are 52939 verified users who constitute an average rating of 4.738.


<font color="darkgrey"> Literature review </font>
The aim of this chapter is to provide readers with an understanding of the different approaches that have been developed in recent years to address the problem of predicting a rating from its text review.
The dataset ‘Digital_Music_5’ has been referenced from [​link​]. The dataset has been indirectly scraped from the online shopping portal Amazon. The data has been collected when buyers shop digital music online on the website. The company Amazon itself uses Machine Learning Algorithms to accurately identify buyer’s music preferences and customize their digital music library by providing recommendations.
The literature [​link​] talks about how it is necessary to predict ratings on the basis of reviews because in today’s fast era, everyone wants to buy products on the basis of reviews but doesn’t have time to read them all. They use three distinct approaches - binary classification, aiming at predicting the rating of a review as low or high, multi-class classification and logistic regression. They use three different state-of-the-art classifiers- Naïve Bayes, Support Vector Machine and Random Forest which are trained and tested where Naïve Bayes and SVM win the game. Additionally, their approach enables users’ feedback to be automatically expressed on a numerical scale.
The literature [​link​] talks about finding whether earlier reviews tend to receive higher helpful ratings because of the duration of the review, instead of the review’s content. To do that, they used favorable and total votes against the reviewer index. To show that whether earlier reviews receive more favorable votes, time series plot is created as a change in total votes of a book can be interpreted in a change in time. Based on the regression models and visualization plots, the dataset showed a trend that earlier reviews receive more favorable votes, but since some slight deviations exist, the possibilities that other factors are controller the number of favorable votes cannot be neglected.
Other literature that identifies user review ratings based on sentiment analysis techniques using a bag-of-words model is [​link​]. The bag-of-words model is the new state-of-the-art approach which makes efficient use of the NLTK library. They have included models that utilize unigrams and bigrams on video game reviews which worked well for them. They also included time-based models that utilized the time a user reviewed a product (year, month, day). They did not serve as good predictors because the variance in the average rating between each year, month, or day was relatively small.
Our model is valuable and different from others as we are approaching the rating prediction with the Latent Factor model which is resulting in a low MSE.


<font color="darkgrey"> Methodology </font>
4.1. Model Evaluation
In order to predict ratings for each transaction, we built four models based on the similarities among items, the significant factors of customers and items, or the regulation in review texts. With a split ratio of 4:1:1 for the training, validation, and test datasets, we trained our models on the training data and selected the optimal hyper-parameters based on validation Mean Squared Error (MSE). Then we apply our four models on the test dataset and computed the test MSEs, to compare the performance of those models.
4.2. Baseline Model
Since most of the ratings in the dataset are 5, we build a baseline model that uses the average rating as a constant predictor of the rating. The average rating is 4.7026, calculated from our training test of 40,000 user reviews. If we keep predicting every rating as the average rating in the training set, the MSE would be 0.47, which is the base MSE that we are trying to beat in our following predictive models.
4.3 Models
4.3.1 Jaccard similarity-based model
Considering that a customer is probably to have a preference for similar types of products and a product is likely to attract a bunch of customers who are similar in some ways, computing the Jaccard similarities between products using Formula 1
...... ​Formula 1 ...... ​ Formula 2
may help with the prediction of ratings of a new product by the same customer. In this model, used features are features of users such as personal information and purchasing habits and features of items such as the ratings and the music types.
We built a Jaccard similarity-based model and trained it with the training dataset. More specifically, we computed the similarities between every two items and appended the values into a list. Then using the similarities as weights, we predict the ratings of the items given by a customer according to the weighted ratings of other items. The MSE of the training set we got is 0.4743. When applying the model to the test dataset, the MSE of the test set is 0.447.
4.3.2 Latent factor model
Since it is hard to figure out what specific factors contribute to customers’ satisfaction, latent semantic analysis may work in this situation. Therefore, we constructed a couple of latent factor models, where no specific feature is used but the latent feature spaces of customers and items are considered instead.
Normalized simple latent factor model
Rating = global average + average rating for a user + average rating for an item
Initially, we built a simple latent factor model, including an global average rating of the user-product pair and two bias terms based on user and product. These bias terms reflect the previous behavior of users and the relative performance of products. Since there are users who are overly positive and there are users who are overly negative, we normalized the ratings by taking the average rating per user and average rating per item. This normalized simple model performs pretty well and has a MSE as low as 0.378.
Thorough latent factor model
Wondering whether additional terms like Gammas can make our model more accountable, we added GammaU and GammaI into the previous latent factor model and formed a more thorough latent factor model (Formula 3).
..... ​Formula 3
To implement this idea, we used another way to train our model instead of finding the optimal parameters
with the help of function scipy.optimize.fmin_l_bfgs_b. With the inner product of Gammas added into our
model, we trained it and selected the best parameters through the optimization process by iterating. We wrote a cost function based on Formula 4 to get the optimal alpha and betas by minimizing C. To do this, we set some convergence conditions to find the optimal Lagrange coefficient lambdas within the ranges we preset (lambda1 for the user and lambda2 for the item.
...... ​Formula 4 When applying the trained model with tuned parameters on our test set, the MSE is 0.349.
4.3.3 Text-analysis based model
Ratings=​θ​0​+​θ​1*​ ​[CountOfMostPopularWords]
In this model, we used linear regression using the 2000 most common unigrams. The 2000 most common word was chosen from a word set that is removed from stop words1 and punctuation based on our train set. We used 2000 as a threshold because of the limited computation environment and because it performs better than models with a threshold of 1000 or 1500. Then for each review in the whole dataset, we transformed the review text into a word list and for each word in the word list, we assign 1 to its corresponding index in the feature if the word​ is​ in the most 2000 popular word set and assign 0 to its corresponding index in the feature if the word ​is not​ in the most 2000 popular word set.
Using a ratio of 4:1:1, we splitted the dataset, which contains features(X) and labels(y) into train/valid/test set. Then, we used ridge regression to tune the parameter ​ λ​ on the validation set. We tuned ​λ among values of 0.01, 0.1, 1, 10 and 100 and found that the model performs best when λ equals to 100, the corresponding MSE is 0.405. Then we used the same value of λ​ on test set and found that the model performance is satisfactory with a MSE of 0.383.
Ratings=​θ​0​+​θ​1*​ ​[TF-IDFScoreofMostPopularWords]
● TF = (Frequency of a word in the document)/(Total words in the document)
● IDF = Log((Total number of docs)/(Number of docs containing the word)) 1 Stop word set was loaded from ​nltk corpus. We chose “english” as our stop word set.

 ● TF-IDF = TF*IDF
In this model, we apply TF-IDF approach to predict the rating of products. Using the TfidfVectorizer from sklearn, we convert the words in each review to a TF-IDF score to create feature vectors. As the previous model, we also remove punctuation and stop words. In addition, we use 70% as a threshold, keeping out of any word that appears in more than 70% of all the reviews. The reason is that we think a word that is too common in all reviews may only contribute little to our rating, which is unique for each product.
Again, we used ridge regression to tune the parameter ​ λ​ on the validation set. We tuned ​λ among values of 0.01, 0.1, 1, 10 and 100 and found that the model performs best when λ equals to 1, the corresponding MSE is 0.36, more than 20% lower than that of the baseline model. The TF-IDF approach beat the previous simple count model.
4.3.4 Other Models
Rating = θ​ 0​ + θ​ 1*[ length of review] + θ3*[ number of votes] + θ2* [ verified or not]
In this basic linear regression model, we appended the length of a review, the number of votes and hot encoded the verified status. We then tuned the parameter to 0.01, getting the lowest MSE on validation set. However, the test set performance is as high as 0.472, even larger than our baseline model.
    11

 5. Model analysis
The second latent factor model (LFM) was chosen as our final model for the following reasons. First of all, the test MSE of this model is the lowest, which is 0.349. Also, the exploratory analysis on the dataset indicates that it is not sparse, thus LFM is supposed to perform well for our prediction tasks. Moreover, considering that the music industry changes fast nowadays, it is hard to track the specific features of all music as well as user preferences. However, LFM can figure out some latent features of similarities automatically, which can be applied to make predictions and recommendations.
To optimize this model, the main thing we need to do is tuning the parameters. Since there is no need to select features to use in LFM and parameters such as alpha and betas are automatically computed through the iteration, we can find the optimal lambdas in the cost function (Formula 4) by looping through a set of values. In this predictive task, we initially set two ranges from 1 to 10 and from 0.2 to 2, and in both cases the best lambda sets are the lowest values. Thus we put a couple of numbers from 0.001 to 10 into two vectors and loop lambdas through them, among which (0.001, 0.01) performed the best. Then we used these tuned parameters to conduct our prediction tasks. However, due to the limitation of our laptop computation capabilities, the lambdas we found might not be the globally optimal ones, this is an issue that we are concerned with. But they could perform relatively well, resulting in a low MSE and no problem such as overfitting.
Before making the decision to use this model, we compared it with those models presented in the previous section. Based on the results, Jaccard similarity-based model could not work well, and a possible interpretation is that Jaccard similarities are calculated with a binary mechanism but ratings are not either 0 or 1. For another thing, the text-based analysis models we built also perform well, with test MSEs of 0.383 and 0.36. Since the difference between the MSEs is tiny, we prefer LFM because text-analysis based models can only predict ratings for existing transactions, which is a major limitation. Although LFM also has some weaknesses such as cold start problems and time-consuming problems which make it hard to make predictions for new customers and items or make real-time predictions, these problems are relatively easy to solve with some more complicated processing on the models.
12

 6. Results and Conclusion
 Based on our results, we realize that the text-analysis based models generally have good performance in terms of test MSE values, but it has major limitations in application since it can only be useful when there are review texts. Although latent factor models also have some limitations, for example, they cannot make good predictions for users and items with insufficient records, they work well in some complicated situations since they do not need to figure out specific features but can consider the joint effect of factors.
In the end, we chose the thorough latent factor model (Model 4) as our final model. In this model, no feature is specifically selected to be used. The parameter alpha is an offset term which equals the average ratings for all users and items. Parameter betas are bias terms, one for the specific user and one for the specific item, and parameter gammas represent the preferences of the customer and the variation of the ratings the item receives. This model stands out because of the compactness of dataset, the fast development of the music industry, and the untraceability of customers’ preferences for music. Additionally, we conducted two models which did not perform well. One is a simple regression model, which fails because it may ignore some significant factors such as the content of the review text. Also, for lots of the reviews, the number of total votes is 0, weakening the efficiency of features. and The other one is a Jaccard similarity-based model, which does not work may be interpreted by the non-duality of ratings.
13

 7. References
● https://pdfs.semanticscholar.org/959b/6d911898ac04dcc706d3d326142d9bbf454b.pdf
● http://snap.stanford.edu/class/cs224w-2012/projects/cs224w-019-final.v01.pdf
● https://github.com/drewmassey/amazon_reviews
● https://pdfs.semanticscholar.org/b71b/fe0fbe009991dc52ac5b03b75b8b44be5aac.pdf
● https://gist.github.com/jbencina/03b2673a6fc27e2717650686b379eeca
● https://www.kaggle.com/c/ugentml16-3/overview
● http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf
● https://medium.com/jbencina/part-1-predicting-amazon-review-ratings-with-text-analytics-i
n-python-fa7c14e91464
● https://www.aclweb.org/anthology/C10-1103.pdf
          14












Simulation generates approximate answers and there is some degree of
error in a quantity estimated by Monte Carlo simulation. In this blog,
we are going to investigate the relationship between then number of
replicates and simulation error.

<font color="darkgrey"> Definition </font>
------------------------------------------

*p̂*:the probability estimated from simulation

p: the true underlying probability

absolute error = \|*p̂* − *p*\|

relative error = \|*p̂* − *p*\|/*p*

<font color="darkgrey">14 X 5 Factorial Experiment Simulation</font>
--------------------------------------------------------------------

replicate number: (2<sup>2</sup>, 2<sup>3</sup>, …, 2<sup>15</sup>)

probability: (0.01, 0.05, 0.10, 0.25, 0.50)

We first generate matrix for each combination of replicate number and
probability, then perform a 14 X 5 factorial experiment simulation

``` r
#generate matrix for each combination of replicate number and probability
abs_error=matrix(NA,14,5)
rel_error=matrix(NA,14,5)
p=c(0.01, 0.05, 0.10, 0.25, 0.50)
#perform a 14 X 5 factorial experiment simulation
for (i in 1:14){
  for (j in 1:5){
    p_hat=rbinom(10000,2^(i+1),p[j])/(2^(i+1))
    abs_error[i,j]=mean(abs(p_hat-p[j]))
    rel_error[i,j]=mean(abs(p_hat-p[j])/p[j])
  }
}
```

<font color="darkgrey"> Figure: Absolute Error </font>
------------------------------------------------------

``` r
x_axis=c("4","8","16","32","64","128","256","512","1024","2048","4096","8192","16384","32768")
prob=c("0.01","0.05","0.10","0.25","0.5")
prob_label=paste0("p=",prob)
plot(as.vector(abs_error[,1]),xaxt='n',type="b",col="red",lwd=3,pch=20,xlim=c(0,14),ylim=c(0,0.2),xlab="N(log2 scale)",ylab="Absolute Error")
axis(1,at=1:14,labels=x_axis,las=2)
lines(as.vector(abs_error[,2]),type="b",col="blue",lwd=3,pch=20)
lines(as.vector(abs_error[,3]),type="b",col="green",lwd=3,pch=20)
lines(as.vector(abs_error[,4]),type="b",col="purple",lwd=3,pch=20)
lines(as.vector(abs_error[,5]),type="b",col="orange",lwd=3,pch=20)
text(1,abs_error[1,],prob_label,pos=2,cex=0.7)
```

![](hw2_files/figure-markdown_github/unnamed-chunk-3-1.png)

The absolute error declines a lot when N is between 4 and 64 and stays
very samll after N exceeds 8192.

The degree of absolute error gets smaller as the number of simulation
replicates increases.

For p=0.01, its original absolute error’s value is the smallest among
five probabilities.

<font color="darkgrey"> Figure: Relative Error </font>
------------------------------------------------------

``` r
x_axis=c("4","8","16","32","64","128","256","512","1024","2048","4096","8192","16384","32768")
prob=c("0.01","0.05","0.10","0.25","0.5")
prob_label=paste0("p=",prob)
plot(as.vector(rel_error[,1]),xaxt='n',type="b",col="red",lwd=3,pch=20,xlim=c(0,14),ylim=c(0,2),xlab="N(log2 scale)",ylab="Relative Error")
axis(1,at=1:14,labels=x_axis,las=2)
lines(as.vector(rel_error[,2]),type="b",col="blue",lwd=3,pch=20)
lines(as.vector(rel_error[,3]),type="b",col="green",lwd=3,pch=20)
lines(as.vector(rel_error[,4]),type="b",col="purple",lwd=3,pch=20)
lines(as.vector(rel_error[,5]),type="b",col="orange",lwd=3,pch=20)
text(1,rel_error[1,],prob_label,pos=2,cex=0.7)
```

![](hw2_files/figure-markdown_github/unnamed-chunk-4-1.png)

The reletive error declines a lot when N is between 4 and 64 and stays
very samll after N exceeds 8192.

The degree of relative error gets smaller as the number of simulation
replicates increases.

For p=0.5, its original relative error’s value is the smallest among
five probabilities.

<font color="darkgrey"> Figure: Absolute Error (with the y-axis is on the log10 scale) </font>
----------------------------------------------------------------------------------------------

Now let’s take log10 for absolute error and generate a new plot

``` r
x_axis=c("4","8","16","32","64","128","256","512","1024","2048","4096","8192","16384","32768")
prob=c("0.01","0.05","0.10","0.25","0.5")
prob_label=paste0("p=",prob)
plot(log10(as.vector(abs_error[,5])),xaxt='n',type="b",col="red",lwd=3,pch=20,xlim=c(0,14),ylim=c(-3.5,-0.5),xlab="N(log2 scale)",ylab="log10(Absolute Error)")
axis(1,at=1:14,labels=x_axis,las=2)
lines(log10(as.vector(abs_error[,4])),type="b",col="blue",lwd=3,pch=20)
lines(log10(as.vector(abs_error[,3])),type="b",col="green",lwd=3,pch=20)
lines(log10(as.vector(abs_error[,2])),type="b",col="purple",lwd=3,pch=20)
lines(log10(as.vector(abs_error[,1])),type="b",col="orange",lwd=3,pch=20)
text(1,log10(abs_error[1,]),prob_label,pos=2,cex=0.7)
```

![](hw2_files/figure-markdown_github/unnamed-chunk-5-1.png)

We can see that it seems to be a linear relationship between
log10(absolute error) and N(log2 scale).

<font color="darkgrey"> Figure: Relative Error (with the y-axis is on the log10 scale) </font>
----------------------------------------------------------------------------------------------

Let’s also take log10 for relative error and generate a new plot

``` r
x_axis=c("4","8","16","32","64","128","256","512","1024","2048","4096","8192","16384","32768")
prob=c("0.01","0.05","0.10","0.25","0.5")
prob_label=paste0("p=",prob)
plot(log10(as.vector(rel_error[,1])),xaxt='n',type="b",col="red",lwd=3,pch=20,xlim=c(0,14),ylim=c(-2.5,0.5),xlab="N(log2 scale)",ylab="log10(Relative Error)")
axis(1,at=1:14,labels=x_axis,las=2)
lines(log10(as.vector(rel_error[,2])),type="b",col="blue",lwd=3,pch=20)
lines(log10(as.vector(rel_error[,3])),type="b",col="green",lwd=3,pch=20)
lines(log10(as.vector(rel_error[,4])),type="b",col="purple",lwd=3,pch=20)
lines(log10(as.vector(rel_error[,5])),type="b",col="orange",lwd=3,pch=20)
text(1,log10(rel_error[1,]),prob_label,pos=2,cex=0.7)
```

![](hw2_files/figure-markdown_github/unnamed-chunk-6-1.png)

We can see that it also seems to be a linear relationship between
log10(relative error) and N(log2 scale).

<font color="darkgrey"> Conclusion </font>
------------------------------------------

From the four graphs, we can see that our intuition is right. That is,
the degree of error gets smaller as the number of simulation replicates
increases.

And for log 10 scale, the figures of the absolute error and relative
error shows that there is a linear relationship between absolute
error/relative error(log10 scale) and N(log2 sacale).

For p=0.01, its original absolute error’s value is the smallest among
five probabilities.

For p=0.5, its original relative error’s value is the smallest among
five probabilities.
