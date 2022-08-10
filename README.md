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
- Shape of dataset is (5572,2)

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








