
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook I try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[6]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# ### Load Data From CSV File  

# In[7]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[24]:


df.shape


# In[25]:


df.dtypes


# ### Convert to date time object 

# In[26]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[27]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[28]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[29]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[31]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[118]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[33]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[34]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[35]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[36]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[37]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[39]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[40]:


X = Feature
X[0:5]


# What are our lables?

# In[178]:


y = df['loan_status'].values
y[0:5]


# In[179]:


# Import label encoder 
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

# Encode labels in column 'Species'. 
y= label_encoder.fit_transform(y)
y


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[180]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[181]:


#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)


# In[182]:


from sklearn.neighbors import KNeighborsClassifier
neighbors = np.arange(1,100)
train_accuracy =np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    #Fit the model
    knn.fit(X, y)
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X, y)


# In[183]:


#KNN_Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[184]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X,y)


# # Decision Tree

# In[185]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X,y)


# # Support Vector Machine

# In[186]:


from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X, y) 


# # Logistic Regression

# In[187]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X,y) 


# # Model Evaluation using Test set

# In[188]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# ### Load Test set for evaluation 

# In[189]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[190]:


#Preprosseing Data:
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])


# In[191]:


test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['dayofweek']


# In[205]:


test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df


# In[206]:


Feature_test = test_df[['Principal','terms','age','Gender','weekend']]
Feature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)
Feature_test.drop(['Master or Above'], axis = 1,inplace=True)


# In[208]:


x_test=Feature_test


# In[209]:


x_test=preprocessing.StandardScaler().fit(x_test).transform(x_test)
x_test[1:5]


# In[210]:


y_test=test_df['loan_status'].values
# Import label encoder 
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

# Encode labels in column 'Species'. 
y_test= label_encoder.fit_transform(y_test) 
y_test


# In[212]:


#prediction
y_pred_1 = knn.predict(x_test)
y_pred_2 = dt.predict(x_test)
y_pred_3 = clf.predict(x_test)
y_pred_4 = lr.predict(x_test)


# # KNN Evaluation:

# In[213]:


print(jaccard_similarity_score(y_test,y_pred_1))


# In[214]:


print(f1_score(y_test,y_pred_1,average=None))


# # Decision Tree Evaluation:

# In[215]:


print(jaccard_similarity_score(y_test,y_pred_2))


# In[216]:


print(f1_score(y_test,y_pred_2,average=None))


# # SVM Evaluation:

# In[217]:


print(jaccard_similarity_score(y_test,y_pred_3))


# In[218]:


print(f1_score(y_test,y_pred_3,average=None))


# # LogisticRegression Evalaution:

# In[219]:


print(jaccard_similarity_score(y_test,y_pred_4))


# In[220]:


print(f1_score(y_test,y_pred_4,average=None))


# In[221]:


# Import label encoder 
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

# Encode labels in column 'Species'. 
y_test= label_encoder.fit_transform(y_test) 


# In[222]:


print(log_loss(y_test,y_pred_4))


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
