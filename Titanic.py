#!/usr/bin/env python
# coding: utf-8

# # Importing packages

# In[ ]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[2]:


train_df = pd.read_csv('train.csv')  # rain data
test_df = pd.read_csv('test.csv') #test data


# In[3]:


train_df.head() # first 5 columns of train data


# In[4]:


# size of train data

train_df.shape


# In[5]:


# size of test data

test_df.shape


# # visualization of data

# In[6]:


train_df['Pclass'].value_counts(normalize = True).plot.bar()


# In[7]:


train_df['Survived'].value_counts(normalize = True).plot.bar()


# In[8]:


survived_class = pd.crosstab(train_df['Pclass'],train_df['Survived'])
survied_class_div = survived_class.div(survived_class.sum(1).astype('float'),axis= 0)


# In[9]:


survied_class_div


# In[10]:


survied_class_div.plot.bar(stacked = True)


# In[11]:


train_df.isnull().sum()


# In[12]:


train_df['Age'].fillna(train_df['Age'].mean(),inplace = True)


# In[13]:


train_df['Cabin'].fillna(train_df['Cabin'].mode()[0],inplace = True)


# In[14]:


bins = [0,18,30,60,100]
groups = ['child','youth','mature','old']
age_bin = pd.cut(train_df['Age'],bins,labels = groups)


# In[15]:


cros  = pd.crosstab(train_df['Survived'],age_bin)


# In[16]:


cros_div = cros.div(cros.sum(1).astype(float),axis = 0)


# In[17]:


cros_div*100


# In[18]:


cros_div.plot.bar(stacked  = True)


# In[19]:


train_df['Sex'].value_counts(normalize = True).plot.bar()


# In[20]:


sex_sur = pd.crosstab(train_df['Survived'],train_df['Sex'])
sex_sur_div = sex_sur.div(sex_sur.sum(1).astype(float),axis = 0)


# In[21]:


sex_sur_div


# In[22]:


sex_sur_div.plot.bar(stacked = True)


# In[23]:


train_df['SibSp'].value_counts(normalize = True).plot.bar()


# In[24]:


SibSp_sur = pd.crosstab(train_df['Survived'],train_df['SibSp'])
SibSp_div = SibSp_sur.div(SibSp_sur.sum(1).astype(float),axis = 0)


# In[25]:


SibSp_div * 100


# In[26]:


SibSp_div.plot.bar()


# In[27]:


Parch_sur = pd.crosstab(train_df['Survived'],train_df['Parch'])
Parch_div = Parch_sur.div(Parch_sur.sum(1).astype(float),axis = 0)


# In[28]:


Parch_div * 100


# In[29]:


Parch_div.plot.bar()


# In[30]:


matrix = train_df.corr()
sns.heatmap(matrix,vmax = .8,square = True,cmap = 'BuPu')


# # Dealing with missing data

# In[31]:


test_df.isnull().sum()


# In[32]:


test_df['Fare'].fillna(test_df['Fare'].mode()[0],inplace = True)
test_df['Cabin'].fillna(test_df['Cabin'].mode()[0],inplace = True)


# In[33]:


test_df['Age'].fillna(test_df['Age'].mean(),inplace = True)


# In[34]:


train= train_df.drop('PassengerId',axis = 1)
test = test_df.drop('PassengerId',axis = 1)


# In[35]:


train.shape


# In[36]:


test.shape


# # creating model 

# In[37]:


train_feature = train_df.drop(['PassengerId','Survived','Name','Ticket','Fare','Cabin','Embarked'],axis = 1 )


# In[38]:


test_feature = test_df.drop(['PassengerId','Name','Ticket','Fare','Cabin','Embarked'],axis = 1 )


# In[39]:


#Input data

X = train_feature

#output data

Y = train.Survived


# In[40]:


#size of input data

X.shape


# In[41]:


#creating dummies 

X = pd.get_dummies(X)
train = pd.get_dummies(train_feature)
test = pd.get_dummies(test_feature)


# In[42]:


#size of input data after creating dummies of input data

X.shape


# In[43]:


#size of train data

train.shape


# In[44]:


#size of test data

test.shape


# In[45]:


# first 5 columns of input data

X.head()


# In[46]:


# first 5 columns of test data

test.head()


# In[48]:


#spliting data 

x_train,x_cv,y_train,y_cv = train_test_split(X,Y,test_size = 0.3)


# In[49]:


model = LogisticRegression()


# In[50]:


#fit train data to model

model.fit(x_train,y_train)


# In[51]:


#predict on test data

predit_lg = model.predict(x_cv)


# In[52]:


# check accuracy of model

accuracy_score(y_cv,predit_lg)


# In[53]:


pred_test = model.predict(test)


# In[54]:


pred_test.shape


# In[55]:


test.shape


# # creating csv file

# In[56]:


#importing sample submission.csv

submission = pd.read_csv('gender_submission.csv')


# In[57]:


#putting value of pred_test in submission 

submission['Survied']  = pred_test


# In[58]:


#putting passenger id of test data in submission data

submission['PassengerId'] = test_df['PassengerId']


# In[59]:


#exporting submission as csv file

pd.DataFrame(submission,columns = ['PassengerId','Survived']).to_csv('prediction.csv')

