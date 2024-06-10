#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

import re
import contractions
from bs4 import BeautifulSoup

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# In[2]:


# get_ipython().system(" pip install bs4 # in case you don't have it installed")
# get_ipython().system(' pip install contractions')
# COMMENTED ABOVE TWO PIP INSTALLS ONLY IN THE .py FILE, BECAUSE IT GIVES ME AN ERROR IF I RUN IT ON MY LOCAL MACHINE,
# JUYPTER NOTEBOOK HAS THESE TWO INSTALLS BECAUSE IT DOES NOT BREAK MY CODE.

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz


# ## Read Data

# In[3]:


#Reading the review and rating data using pandas read_table function
col = ['review_body', 'star_rating']
data = pd.read_table('amazon_reviews_us_Jewelry_v1_00.tsv', usecols = col)


# ## Keep Reviews and Ratings

# In[4]:


# This step is completed in the above function as a parameter.


#  ## We select 20000 reviews randomly from each rating class.
# 
# 

# In[5]:


#removing all Nan values in the rating
data['star_rating'] = data['star_rating'].fillna(0)

#Dropping all the rating that is not 1-5 rating (5 classes)
data.drop(data[(data['star_rating'] != 1) & (data['star_rating'] != 2) & (data['star_rating'] != 3) & (data['star_rating'] != 4) & (data['star_rating'] != 5)].index, inplace=True)
types = data['star_rating'].unique()

#Grouping the data by the ratings 1-5
new_data = data.groupby('star_rating')

#split each class as a dataframe
group_1 = new_data.get_group(1)
group_2 = new_data.get_group(2)
group_3 = new_data.get_group(3)
group_4 = new_data.get_group(4)
group_5 = new_data.get_group(5)

#randomize 20000 reviews from each class
group_1 = group_1.sample(n=20000)
group_2 = group_2.sample(n=20000)
group_3 = group_3.sample(n=20000)
group_4 = group_4.sample(n=20000)
group_5 = group_5.sample(n=20000)

#combine all the data then randomize again
reduced_data = group_1.append(group_2)
reduced_data = reduced_data.append(group_3)
reduced_data = reduced_data.append(group_4)
reduced_data = reduced_data.append(group_5)
reduced_data = reduced_data.sample(n=100000)


# # Data Cleaning
# 
# 

# In[6]:


# Calculatig the average before data cleaning
average_before = reduced_data['review_body'].str.len()
print('The Average Length of the Reviews Before Cleaning: ')
print(average_before.mean())

# Making all characters lower-case
reduced_data['review_body'] = reduced_data['review_body'].str.lower()

# Removing all extra white spaces
reduced_data['review_body'] = reduced_data['review_body'].str.strip()

# Removing all the HTML code using Regex, this will remove all the tag that open with < and close with >, including everything in-between
reduced_data['review_body'] = reduced_data['review_body'].str.replace('<[^<]+?>', '')

# Removing all URL links using Regex, this will remove all links that start with http: and/or www.
reduced_data['review_body'] = reduced_data['review_body'].str.replace('http\S+|www.\S+', '')

# Removing all non-alphabetical (not a-z or A-Z) characters and replacing them with space
reduced_data['review_body'] = reduced_data['review_body'].str.replace('[^a-zA-Z\s]', '')

# Casting all review data as a string to make sure the data has no errors
reduced_data['review_body'] = reduced_data['review_body'].astype('str')

# Using the contractions library, we apply the "contraction.fix" function to every review in our database using lambda
reduced_data["review_body"] = reduced_data['review_body'].apply(lambda x: contractions.fix(x))

#Calculatig the average after data cleaning
average_after = reduced_data['review_body'].str.len()
print('The Average Length of the Reviews After Cleaning: ')
print(average_after.mean())


# # Pre-processing

# ## remove the stop words 

# In[7]:


# Calculating the average characters of the review before pre-processing
average_before = reduced_data['review_body'].str.len()
print('The Average Length of the Reviews Before Pre-Processing: ')
print(average_before.mean())

# Using the NLTK stopwords library, we get the English stopwords
from nltk.corpus import stopwords
stopw = stopwords.words('english')
# For every one of the reviews in the dataset, we split the string into individual words, we then verify if any of them is a stop word,
# if not, we can concatenate it back to the review. In this process, we remove all the stopwords. 
reduced_data['review_body'] = reduced_data['review_body'].apply(lambda x: ' '.join([i for i in x.split() if i not in (stopw)]))


# ## perform lemmatization  

# In[8]:


# Using the NLTK library, we get the Lemmatizer
from nltk.stem import WordNetLemmatizer

# For every one of the reviews in the dataset, we split the string into individual words, we then apply the lemmatizer for every word and then concatenate them back.
reduced_data['review_body'] = reduced_data['review_body'].apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(i) for i in x.split()]))


# In[9]:


# Calculating the average characters of the review after pre-processing
average_after = reduced_data['review_body'].str.len()
print('The Average Length of the Reviews After Pre-Processing: ')
print(average_after.mean())


# # TF-IDF Feature Extraction

# In[10]:


# Using the sklearn feature extraction library we create a TF-IDF Vectorizer and extract features
feature_vec = TfidfVectorizer()
features = feature_vec.fit_transform(reduced_data['review_body'])

# We create a numpy array to store all the labels for later use
labels = reduced_data['star_rating'].values


# In[11]:


# We create a vector to store the names/words of each feature that we got
names = feature_vec.get_feature_names()

# We use the function train_test_split to split all the features and labels into training and testing counterparts
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=0.8)


# In[12]:


# We cast all the labels as integers because some of the review are floats (like 1.0) instead of integers (like 1)
train_int_labels = train_labels.astype(int)
test_int_labels = test_labels.astype(int)


# ## Perceptron

# In[13]:


# We create a perceptron model instance
perceptron_model = Perceptron()

# We train the model using the training features and labels
perceptron_model.fit(train_features, train_int_labels)


# In[14]:


# We get the accuarcy score using test features and labels
print("The Accuracy Score for Perceptron is: ")
Perc_acc = perceptron_model.score(test_features, test_int_labels)
print(Perc_acc)


# In[15]:


# We predict the ouput labels by using the test data
p_test_pred = perceptron_model.predict(test_features)

# We obtain the precision, recall and F1 scores using the sklearn metrics library
precision_mark_p = precision_score(test_int_labels, p_test_pred, average=None)
recall_mark_p = recall_score(test_int_labels, p_test_pred, average=None)
f1_mark_p = f1_score(test_int_labels, p_test_pred, average=None)


# In[16]:


# We create arrays to store all the score values
prec_arr = []
recall_arr = []
f1_arr = []
avg_arr = []

# The average of precision, recall and F1 scores are calculated by summing them individually and divide by the total number of classes (which is 5)
avg_arr = [sum(precision_mark_p)/5, sum(recall_mark_p)/5, sum(f1_mark_p)/5]

# Converting the float scores into strings
for x in precision_mark_p:
    prec_arr.append(str(x))
    
for x in recall_mark_p:
    recall_arr.append(str(x))

for x in f1_mark_p:
    f1_arr.append(str(x))

# Organizing the string outputs into 5 classes and the average
c_one = [prec_arr[0], recall_arr[0], f1_arr[0]]
c_two = [prec_arr[1], recall_arr[1], f1_arr[1]]
c_three = [prec_arr[2], recall_arr[2], f1_arr[2]]
c_four = [prec_arr[3], recall_arr[3], f1_arr[3]]
c_five = [prec_arr[4], recall_arr[4], f1_arr[4]]
c_avg = ' '.join(str(x) for x in avg_arr)

# Printing the scores and averages from the Perceptron Model
print("In the Perceptron Model:" )
c_one = ', '.join(c_one)
print("The Scores Class 1 are: " + c_one)
c_two = ', '.join(c_two)
print("The Scores Class 2 are: " + c_two)
c_three = ', '.join(c_three)
print("The Scores Class 3 are: " + c_three)
c_four = ', '.join(c_four)
print("The Scores Class 4 are: " + c_four)
c_five = ', '.join(c_five)
print("The Scores Class 5 are: " + c_five)

print("The Averages of the Scores: " + c_avg)
print("* Scores are in the order of Precision, Recall, F1")


# # SVM

# In[17]:


# We create a linear SVM model instance
svm_linear_model = svm.LinearSVC()

# We train the model using the training features and labels
svm_linear_model.fit(train_features, train_int_labels)


# In[18]:


# We get the accuarcy score using test features and labels
print("The Accuracy Score for SVM is: ")
svm_acc = svm_linear_model.score(test_features, test_int_labels)
print(svm_acc)


# In[19]:


# We predict the ouput labels by using the test data
svm_test_pred = svm_linear_model.predict(test_features)

# We obtain the precision, recall and F1 scores using the sklearn metrics library
precision_mark_svm = precision_score(test_int_labels, svm_test_pred, average=None)
recall_mark_svm = recall_score(test_int_labels, svm_test_pred, average=None)
f1_mark_svm = f1_score(test_int_labels, svm_test_pred, average=None)


# In[20]:


# We create arrays to store all the score values
prec_arr_svm = []
recall_arr_svm = []
f1_arr_svm = []
avg_arr_svm = []

# The average of precision, recall and F1 scores are calculated by summing them individually and divide by the total number of classes (which is 5)
avg_arr_svm = [sum(precision_mark_svm)/5, sum(recall_mark_svm)/5, sum(f1_mark_svm)/5]

# Converting the float scores into strings
for x in precision_mark_svm:
    prec_arr_svm.append(str(x))
    
for x in recall_mark_svm:
    recall_arr_svm.append(str(x))

for x in f1_mark_svm:
    f1_arr_svm.append(str(x))
    
# Organizing the string outputs into 5 classes and the average
c_one_svm = [prec_arr_svm[0], recall_arr_svm[0], f1_arr_svm[0]]
c_two_svm = [prec_arr_svm[1], recall_arr_svm[1], f1_arr_svm[1]]
c_three_svm = [prec_arr_svm[2], recall_arr_svm[2], f1_arr_svm[2]]
c_four_svm = [prec_arr_svm[3], recall_arr_svm[3], f1_arr_svm[3]]
c_five_svm = [prec_arr_svm[4], recall_arr_svm[4], f1_arr_svm[4]]
c_avg_svm = ' '.join(str(x) for x in avg_arr_svm)

# Printing the scores and averages from the SVM Model
print("In the SVM Model:")
c_one_svm = ', '.join(c_one_svm)
print("The Scores Class 1 are: " + c_one_svm)
c_two_svm = ', '.join(c_two_svm)
print("The Scores Class 2 are: " + c_two_svm)
c_three_svm = ', '.join(c_three_svm)
print("The Scores Class 3 are: " + c_three_svm)
c_four_svm = ', '.join(c_four_svm)
print("The Scores Class 4 are: " + c_four_svm)
c_five_svm = ', '.join(c_five_svm)
print("The Scores Class 5 are: " + c_five_svm)

print("The Averages of the Scores: " + c_avg_svm)
print("* Scores are in the order of Precision, Recall, F1")


# # Logistic Regression

# In[21]:


# We create a Logistic Regression model instance
LR_model = LogisticRegression(max_iter=1000)

# We train the model using the training features and labels
LR_model.fit(train_features, train_int_labels)


# In[22]:


# We get the accuarcy score using test features and labels
print("The Accuracy Score for Logistic Regression is: ")
LR_acc = LR_model.score(test_features, test_int_labels)
print(LR_acc)


# In[23]:


# We predict the ouput labels by using the test data
LR_test_pred = LR_model.predict(test_features)

# We obtain the precision, recall and F1 scores using the sklearn metrics library
precision_mark_LR = precision_score(test_int_labels, LR_test_pred, average=None)
recall_mark_LR = recall_score(test_int_labels, LR_test_pred, average=None)
f1_mark_LR = f1_score(test_int_labels, LR_test_pred, average=None)


# In[24]:


# We create arrays to store all the score values
prec_arr_LR = []
recall_arr_LR = []
f1_arr_LR = []
avg_arr_LR = []

# The average of precision, recall and F1 scores are calculated by summing them individually and divide by the total number of classes (which is 5)
avg_arr_LR = [sum(precision_mark_LR)/5, sum(recall_mark_LR)/5, sum(f1_mark_LR)/5]

# Converting the float scores into strings
for x in precision_mark_LR:
    prec_arr_LR.append(str(x))
    
for x in recall_mark_LR:
    recall_arr_LR.append(str(x))

for x in f1_mark_LR:
    f1_arr_LR.append(str(x))
    
# Organizing the string outputs into 5 classes and the average
c_one_LR = [prec_arr_LR[0], recall_arr_LR[0], f1_arr_LR[0]]
c_two_LR = [prec_arr_LR[1], recall_arr_LR[1], f1_arr_LR[1]]
c_three_LR = [prec_arr_LR[2], recall_arr_LR[2], f1_arr_LR[2]]
c_four_LR = [prec_arr_LR[3], recall_arr_LR[3], f1_arr_LR[3]]
c_five_LR = [prec_arr_LR[4], recall_arr_LR[4], f1_arr_LR[4]]
c_avg_LR = ' '.join(str(x) for x in avg_arr_LR)

# Printing the scores and averages from the Logisitic Regression Model
print("In the Logistic Regression Model:")
c_one_LR = ', '.join(c_one_LR)
print("The Scores Class 1 are: " + c_one_LR)
c_two_LR = ', '.join(c_two_LR)
print("The Scores Class 2 are: " + c_two_LR)
c_three_LR = ', '.join(c_three_LR)
print("The Scores Class 3 are: " + c_three_LR)
c_four_LR = ', '.join(c_four_LR)
print("The Scores Class 4 are: " + c_four_LR)
c_five_LR = ', '.join(c_five_LR)
print("The Scores Class 5 are: " + c_five_LR)

print("The Averages of the Scores: " + c_avg_LR)
print("* Scores are in the order of Precision, Recall, F1")


# # Naive Bayes

# In[25]:


# We create a Naive Bayes model instance
bayes_model = MultinomialNB()

# We train the model using the training features and labels
bayes_model.fit(train_features, train_int_labels)


# In[26]:


# We get the accuarcy score using test features and labels
print("The Accuracy Score for Naive Bayes is: ")
bayes_acc = bayes_model.score(test_features, test_int_labels)
print(bayes_acc)


# In[27]:


# We predict the ouput labels by using the test data
bayes_test_pred = bayes_model.predict(test_features)

# We obtain the precision, recall and F1 scores using the sklearn metrics library
precision_mark_bayes = precision_score(test_int_labels, bayes_test_pred, average=None)
recall_mark_bayes = recall_score(test_int_labels, bayes_test_pred, average=None)
f1_mark_bayes = f1_score(test_int_labels, bayes_test_pred, average=None)


# In[28]:


# We create arrays to store all the score values
prec_arr_bayes = []
recall_arr_bayes = []
f1_arr_bayes = []
avg_arr_bayes = []

# The average of precision, recall and F1 scores are calculated by summing them individually and divide by the total number of classes (which is 5)
avg_arr_bayes = [sum(precision_mark_bayes)/5, sum(recall_mark_bayes)/5, sum(f1_mark_bayes)/5]

# Converting the float scores into strings
for x in precision_mark_bayes:
    prec_arr_bayes.append(str(x))
    
for x in recall_mark_bayes:
    recall_arr_bayes.append(str(x))

for x in f1_mark_bayes:
    f1_arr_bayes.append(str(x))
    
# Organizing the string outputs into 5 classes and the average
c_one_bayes = [prec_arr_bayes[0], recall_arr_bayes[0], f1_arr_bayes[0]]
c_two_bayes = [prec_arr_bayes[1], recall_arr_bayes[1], f1_arr_bayes[1]]
c_three_bayes = [prec_arr_bayes[2], recall_arr_bayes[2], f1_arr_bayes[2]]
c_four_bayes = [prec_arr_bayes[3], recall_arr_bayes[3], f1_arr_bayes[3]]
c_five_bayes = [prec_arr_bayes[4], recall_arr_bayes[4], f1_arr_bayes[4]]
c_avg_bayes = ' '.join(str(x) for x in avg_arr_bayes)

# Printing the scores and averages from the Multinomial Navie Bayes Model
print("In the Multinomial Naive Bayes Model:")
c_one_bayes = ', '.join(c_one_bayes)
print("The Scores Class 1 are: " + c_one_bayes)
c_two_bayes = ', '.join(c_two_bayes)
print("The Scores Class 2 are: " + c_two_bayes)
c_three_bayes = ', '.join(c_three_bayes)
print("The Scores Class 3 are: " + c_three_bayes)
c_four_bayes = ', '.join(c_four_bayes)
print("The Scores Class 4 are: " + c_four_bayes)
c_five_bayes = ', '.join(c_five_bayes)
print("The Scores Class 5 are: " + c_five_bayes)

print("The Averages of the Scores: " + c_avg_bayes)
print("* Scores are in the order of Precision, Recall, F1")

