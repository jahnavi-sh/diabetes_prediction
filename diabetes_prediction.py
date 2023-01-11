#Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively 
#use the insulin it produces. Insuline is a hormone that regulates blood glucose. High blood sugar level can cause a number of other 
#health problems relating to heart, skin, gut etc. Nowadays, a large population of both young and old are being diagnosed wtih 
#diabetes. 
#Here, we will build a machine learning model that predicts whether a patient as diabetes or not based on the medical reports 
#being given. We use several medical information such as dna of the patient, their glucose level, their blood sugar level etc. 

#workflow 
#1. collect the data 
#2. data prepocessing - This is done because there is a huge amount of medical data which we need to standardize so that 
#   it all lies in the same range and can be easily fed to the machine learning algorithm. 
#3. train test split 
#4. model being used is support vector machine classifier model. 

#load libraries 
import numpy as np         #for linear algebra
import pandas as pd        #for data processing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#we use Diabetes dataset from kaggle for this study. the National Institute of Diabetes and Digestive and Kidney Diseases is the 
#original source of this database.
#data collection 
diabetes_dataset = pd.read_csv(r'')

#first step is to get familiar with the structure of the data

#top 5 rows of the dataset
diabetes_dataset.head()
#all the columns are integers, except BMI and DiabetesPedigreeFunction
#the target variable is the labels with values 0 an 1. 
#0 - patient does not have diabetes 
#1 - patient has diabetes 

#view the number of rows and columns 
diabetes_dataset.shape

#statistical measures of the dataset
diabetes_dataset.describe()

#calculate number of diabetic and non diabetic patients 
diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby['Outcome'].mean()

#splitting the dependent and independent features
#separating data and label 
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

#standardize the data 
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
#all values are between 0 and 1

X = standardized_data

#train and test 
X_train, X_test, Y_train, Y_test - train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print (X.shape, X_train,shape, X_test.shape)

#fitting model 
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

#evaluate model 
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print ('accuracy score of training data', training_data_accuracy)

#evaluating on test data 
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print ('accuracy score of test data', test_data_accuracy)

#making a predictive system 
input_data = ()
input_data_as_an_numpy_array = np.asarray(input_data)

#reshape to predict for one instance 
input_data_reshape = input_data_as_an_numpy_array.reshape(1, -1)

#standardize input data 
std_data = scaler.transform(input_data_reshape)
print (std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0]==0):
    print ("the person is not diabetic")
else:
    print ("the person is diabetic")
