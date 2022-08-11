# Email-Sms-Spam-Classifier

## Objective : Prediction of Spam Email or SMS using Machine Learning

## Introduction : 
### Spam email is unwanted junk email sent out in massive amount or in bulk to an indiscriminate recipient list. Generally, spam is sent for commercial purposes. It is sent in massive volume by botnets, networks of infected computers. Spam email can often be a malicious attempt to gain access to your system. Spam prevents the user from making full and good utilization of cpu time, storage capacity and network bandwidth. It becomes a huge problem especially at times when there are Spam mails which come in between important business mails. Hence, it becomes inevitable to solve such problems which are encountered by spam email. So, this problem can be solved by using Machine Learning methods which can successfully detect and filter spam. 

## Problem Statement : 
### The person responsible for sending the spam messages is referred to as the spammer. Such a person gathers email addresses from different websites, chat rooms etc. The huge volume of Spam mails flowing through the computer networks have destructive effects on the memory space of the email server, communication bandwith, cpu power and user time. In all, existing system does not find spam mails effectively. Hence, it also results in untold financial losses to many users. It leads to low test and prediction accuracy, less security and also loss of data.

## So according to above problem statement, I have built the machine learning model which will predict the Email or SMS is Spam or Not Spam.

### Python libraries used to build the model:
- Numpy 
- Pandas 
- Matplotlib
- Seaborn 
- Scikit-learn

### The dataset contains the two columns:
- TEXT (input)
- Target (output: 1. Spam 2. Not Spam)
- Shape of dataset is (5572 , 2)

## The Model is built by using following steps :
- Data cleaning
- EDA
- Text Preprocessing
- Model building
- Evaluation
- Improvement

### 1. Data Cleaning : 
- I have renamed the columns as column names in the dataset were v1 and v2 and new columns are v1 as 'target' and v2 as 'text'
![image](https://user-images.githubusercontent.com/104545490/183970829-52048745-eec0-4cb2-9748-9da3dcc7c219.png)

- As you can see above, our target variable is a categorical feature so I have converted into numerical variable by using LABEL ENCODING from SKLEARN library. i.e. I have maped the SPAM category as 1 and ham category as 0.
![image](https://user-images.githubusercontent.com/104545490/183972132-5bfa6782-be60-451d-82b8-f09768eb3a4d.png)

- After that I have checked the null values present in the dataset
![image](https://user-images.githubusercontent.com/104545490/183973010-472efc6b-6c39-43a4-b374-df467f46e5d9.png)
I found no missing value in the dataset

- I checked the duplicate values in the dataset
![image](https://user-images.githubusercontent.com/104545490/183973704-98ec68a9-a76d-49d8-a30c-ca405e65dc82.png)
Now shape of the dataset becomes (5169,2) after dropping all duplicate rows

### 2. EDA(Exploratory Data Analysis):
- I have checked the distrbution of ham and spam category in the target variable(i.e. how many are ham and how many are spam). I used the matplotlib library for the visualization in the form of pie chart
![image](https://user-images.githubusercontent.com/104545490/183974807-3062e182-6843-4a89-af0e-ef14e246b728.png)

- After that I have created the 3 columns of number of characters, number of words and number of sentences in every single row or sample. I have used the NLTK which is a standard python library that provides a set of diverse algorithms for NLP. It is one of the most used libraries for NLP and Computational Linguistics.
![image](https://user-images.githubusercontent.com/104545490/183976184-ad1b6e74-4711-4979-b9b3-6f129bc40934.png)

- I have created the Histplot to see how number of characters and number of words distrbuted in input
![image](https://user-images.githubusercontent.com/104545490/183976970-489aa10a-204d-4f4a-bc86-bb1d5e68c847.png)
![image](https://user-images.githubusercontent.com/104545490/183977041-31aa4426-533d-4c2f-9364-7bea6f508d56.png)

- The I checked how number of characters, number of words and number of sentences are correlated by using seaborn library
![image](https://user-images.githubusercontent.com/104545490/183978617-707c0052-95b8-4c6a-a5af-d4bc29fee614.png)
![image](https://user-images.githubusercontent.com/104545490/183978741-de0a5062-a77e-41f5-aa47-eb222e17aabf.png)

### 3. Data Preprocessing:
- Lower case
- Tokenization
- Removing special characters
- Removing stop words and punctuation
- Stemming
#### 1. Lower case: 
I have converted all the words into lower words to not to repeat the words and characters as both meaning is same

#### 2. Tokenization:
Tokenization is the preocess of converting paragraph into list of sentences and sentences into list of words and I converted text data into list of words for every sample row

#### 3. Removing special characters:
The special characters like '!','%','*','$' are removed from sentences

#### 4. Removing Stop words:
The words like 'The', 'is', 'am', 'it' are removed because it does no meaning and does not affects on output

#### 5. Stemming:
Stemming is process of finding the root words of the all words in the text data. For eg. 'calling', 'called' have root word 'call' and 'gone', 'goes' have root word 'go'
#### 6. I have created the wordcloud chart which shows most frequently words when EMAIL or SMS is spam and ham
##### For Spam
![image](https://user-images.githubusercontent.com/104545490/183982163-7a80fda0-0835-4d66-b7f3-7e81483f9f4b.png)
##### For ham
![image](https://user-images.githubusercontent.com/104545490/183982231-59fc96bc-465b-4bcb-9c22-7105764a15be.png)

#### 7. Our final datastet will be:
![image](https://user-images.githubusercontent.com/104545490/183982961-f8759d87-a656-449e-922b-844d92d38091.png)

#### 8. Using TFIDF(term frequency-inverse document frequency) I have converted the text input into vectors after that we wiil get the as many columns as we have unique words in the input dataset
![image](https://user-images.githubusercontent.com/104545490/183986288-9a7636a9-b0c5-4e4c-9677-49c19d4fd859.png)

The shape of dataset after preprocessing becomes
![image](https://user-images.githubusercontent.com/104545490/183986460-2ce30c77-8900-4d7a-88fb-46e56b77f059.png)


### 4. Model Building and Evaluation :

##### As our problem is of text classification so the algorithm called Naive Bayes Classifier works very well on this type of data. i.e. Text Data 
#### Naive Bayes Classifier have 3 types :
- Multinomial Naive Bayes :
Multinomial Naïve Bayes consider a feature vector where a given term represents the number of times it appears or very often i.e. frequency. Multinomial Naive Bayes - Widely used classifier for document classification which keeps the count of frequent words present in the documents.

- Bernoulli Naive Bayes : Bernoulli is a binary algorithm used when the feature is present or not. 

- Guassian Naive Bayes : Gaussian is based on continuous distribution i.e. Used when we are dealing with continuous data.

#### I have calculated the accuracy, confusion matrix and precision score of each of three classifiers
![image](https://user-images.githubusercontent.com/104545490/183988425-13dfd954-1cc8-4075-9bfb-a6ba9f7f9657.png)

#### From three I have got the good accuracy and prescision of Multinomial Naive Bayes which is 97.1 % and precision is 100 % which is best for our model.
#### Then I have tried with different algorithms like Logistic Regression, Decsion Tree, Random Forest Classiifier, ADABOOST, XG-BOOST, KNeighbours Classifier. After that I got the accuracy and precision as follows and among them Multinomial Naive Bayes Classifier performing best:
![image](https://user-images.githubusercontent.com/104545490/183989729-5a26cd2f-88a3-4941-b609-4ac99be0d1ab.png)

#### Lets see demo of model:
- 1. Testing 1
![image](https://user-images.githubusercontent.com/104545490/184197792-a48bee6d-7f64-4abb-a5c9-aee8c53f4f76.png)
 
- 2. Testing 2
![image](https://user-images.githubusercontent.com/104545490/184199655-43681129-5004-4f68-a128-7696dbe9aae0.png)


## Conclusion : 
### This is how I have created the EMAIL/SMS Spam Classifier model by using the machine learning algorithm of Naive Bayes Classifier. 








