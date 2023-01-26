# diabetes_prediction

This document is written to provide aid in understanding the project.

Contents of the document - 
1. Understanding the problem statement 
2. Machine learning 
3. Types of machine learning models with examples 
4. Machine learning algorithm used for the model - Logistic Regression
5. NumPy library 
6. Pandas library 
7. Scikit-learn library 
8. Data preprocessing - StandardScaler
9. Exploratory data analysis 
10. Train-test split 
11. Model evaluation - accuracy 

First we need to understand the problem statement - 

Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose. High blood sugar level can cause a number of other health problems relating to heart, skin, gut etc. Nowadays, a large population of both young and old are being diagnosed with diabetes. 

Here, we will build a machine learning model that predicts whether a patient as diabetes or not based on the medical reports being given. We use several medical information such as dna of the patient, their glucose level, their blood sugar level etc. 

Machine learning - 

Machine learning enables the processing of sonar signals and target detection. Machine Learning is a subset of Artificial Intelligence. This involves the development of computer systems that are able to learn by using algorithms and statistical measures to study data and draw results from it. Machine learning is basically an integration of computer systems, statistical mathematics and data.

Machine Learning is further divided into three classes - Supervised learning, Unsupervised learning and Reinforcement Learning. 

Supervised learning is a machine learning method in which models are trained using labelled data. In supervised learning, models need to find the mapping function and find a relationship between the input and output. In this, the user has a somewhat idea of what the output should look like. It is of two types - regression (predicts results with continuous output. For example, given the picture of a person, we have to predict their age on the basis of the given picture) and classification (predict results in a discrete output. For example, given a patient with a tumor, we have to predict whether the tumor is malignant or benign.) 

Unsupervised learning is a method in which patterns are inferred from the unlabelled input data. It allows us to approach problems with little or no idea what the results should look like. We can derive structure from the data where we don’t necessarily know the effect of variables. We can derive the structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback on the prediction results. It is of two types - clustering (model groups input data into groups that are somehow similar or related by different variables. For example, clustering data of thousands of genes into groups) and non-clustering (models identifies individual inputs. It helps us find structure in a chaotic environment. For example, the cocktail party problem where we need to identify different speakers from a given audiotape.)

Reinforcement learning is a feedback-based machine learning technique. It is about taking suitable action to maximise reward in a particular situation. For example, a robotic dog learning the movement of his arms or teaching self-driving cars how to depict the best route for travelling. 

In this case, we will use Support vector machine classifier model

Svm can be used to handle both classification and regression on linear and non-linear data. It can find complex relationships between data without much data transformation of our own. It’s a great option when working with smaller datasets that have tens to hundreds of thousands of features. It typically find more accurate results when compared to other algorithms because of their ability to handle small, complex datasets. 
SVMs are different from other classification algorithms because it chooses the decision boundary that maximise the distance from the nearest data points of all the classes. This decision boundary is maximum margin classifier or maximum margin hyper plane. 

A simple svm classifier works by making a straight line between two classes. All the data points on one side of the line will represent a category and data points on other side are put into different category. This means there can be infinite number of lines to choose from. 
SVM algorithm chooses the best line to classify data points. It chooses the line that separates data and is further away from the closest data points are possible. This line is the decision boundary. This decision boundary can be a hyperplane which uses multiple number of features. 

SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme points are called support vectors and therefore the algorithm is called support vector machine. 

Libraries used in the project -

NumPy  

It is a python library used for working with arrays. It has functions for working in the domain of linear algebra, fourier transform, and matrices. It is the fundamental package for scientific computing with python. NumPy stands for numerical python. 

NumPy is preferred because it is faster than traditional python lists. It has supporting functions that make working with ndarray very easy. Arrays are frequently used where speed and resources are very important. NumPy arrays are faster because it is stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently. This is locality of reference in computer science. 

Pandas - 

Pandas is made for working with relational or labelled data both easily and intuitively. It provides various data structures and operations for manipulating numerical data and time series. 

It has a lot of advantages like - 
1. Fast and efficient for manipulating and analyzing data
2. Data from different file objects can be loaded 
3. Easy handling of missing data in data preprocessing 
4. Size mutability 
5. Easy dataset merging and joining 
6. Flexible reshaping and pivoting of datasets 
7. Gives time-series functionality 

Pandas is built on top of NumPy library. That means that a lot of structures of NumPy are used or replicated in Pandas. The data produced by pandas are often used as input for plotting functions of Matplotlib, statistical analysis in SciPy, and machine learning algorithms in Scikit-learn. 

Scikit-Learn - 

It provides efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It has numerous machine learning, pre-processing, cross validation, and visualization algorithms. 

Data preprocessing - 

This is done because there is a huge amount of medical data we are dealing with here. So we need to standardize so that it all lies in the same range and can be easily fed to the machine learning algorithm. 
StandardScaler scales the data to a variance of 1 after the mean is reduced to 0. 

Exploratory data analysis -

Exploratory data analysis is the process of performing initial investigation on the data to discover patterns or spot anomalies. It is done to test the hypothesis and to check assumptions with the help of summary statistics and graphical representations. 

‘describe()’ method returns description of data in DataFrame. It tells us the following information for each column - 
Count - number of non-empty values
Mean - the average (mean) value  
Std - standard deviation
Min - minimum value
25% - the 25 percentile 
50% - the 50 percentile 
75% - the 75 percentile
Max - maximum value

Train-test split - 

The entire dataset is split into training dataset and testing dataset. Usually, 80-20 or 70-30 split is done. The train-test split is used to prevent the model from overfitting and to estimate the performance of prediction-based algorithms. We need to split the dataset to evaluate how well our machine learning model performs. The train set is used to fit the model, and statistics of training set are known. Test set is for predictions. 

This is done by using scikit-learn library and train_test_split() function. 
Parameters - 
1. *arrays: inputs such as lists, arrays, data frames, or matrices
2. test_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our test size. its default value is none.
3. train_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our train size. its default value is none.
4. random_state: this parameter is used to control the shuffling applied to the data before applying the split. it acts as a seed.
5. shuffle: This parameter is used to shuffle the data before splitting. Its default value is true.
6. stratify: This parameter is used to split the data in a stratified fashion.

Model evaluation - 

Model evaluation is done to test the performance of machine learning model. It is done to determine whether the model is a good fit for the input dataset or not. 

In this case, we use accuracy. Accuracy is a performance metrics that is used to test a binary classification model. Accuracy measures the proportion of correct prediction to total data points.

Accuracy = ( tp + tn) / ( tp + fp + tn + fn )

- Tp - true positive. This refers to the total number of observations that belong to the positive class and have been predicted correctly. 
- Tn - true negatives. It is total number of observations that belong to the negative class and have been predicted correctly 
- Fp - false positives. It total number of observations that have been predicted to belong to positive class, but instead belong to the negative class. 
- Fn - false negatives. It is total number of observations that have been predicted to be a part of negative class but instead belong to the positive class. 
