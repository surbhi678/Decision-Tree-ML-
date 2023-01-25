#!/usr/bin/env python
# coding: utf-8

# In[24]:

# Part One:Bank 
#Load all required packages
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer as Imputer


# In[25]:


#Load the bank.csv Dataset and set the delimiter

data = pd.read_csv("bank.csv", delimiter=";") 



# In[26]:


# View the first five rows of the dataset
data.head()


# In[27]:


# Obtain some basic statistics about the variables in the dataset
data.info()


# In[28]:


#  Change the unit of 'duration' from seconds to minutes
data['duration'] = data['duration'].apply(lambda n:n/60).round(2)


# In[29]:


# we are finding the percentage of each class in the feature 'y'
class_values = (data['y'].value_counts()/data['y'].value_counts().sum())*100
print(class_values)
#The class distribution in the target is ~89:11. This is a clear indication of imbalance.


# In[30]:


#Examining some statistics about the data 
data.describe() 


# In[31]:


data.median()


# In[32]:


#Create Scatter matrix plots
scatter = pd.plotting.scatter_matrix(data,figsize= (20,20))
plt.show()
#[code adapted from: https://www.marsja.se/pandas-scatter-matrix-pair-plot/]


# In[33]:


# Data Pre-Processing


# In[34]:


# Before pre-processing, make a copy of the data for safety
data1= data.copy()


# In[35]:


#Check for any null values in the dataset
data1[data.isnull().any(axis=1)].count()


# In[36]:


#dropping columns
# data1.drop(['contact','day','month','duration'], inplace=True, axis=1)
data1.drop(['duration'], inplace=True, axis=1)


# In[37]:


print("# Missing value 'job' variable: {0}".format(len(data1.loc[data1['job'] == "unknown"])))
print("# Missing value 'marital' variable: {0}".format(len(data1.loc[data1['marital'] == "unknown"])))
print("# Missing value 'education' variable: {0}".format(len(data1.loc[data1['education'] == "unknown"])))
print("# Missing value 'default' variable: {0}".format(len(data1.loc[data1['default'] == "unknown"])))
print("# Missing value 'contact' variable: {0}".format(len(data1.loc[data1['contact'] == "unknown"])))
print("# Missing value 'day' variable: {0}".format(len(data1.loc[data1['day'] == "unknown"])))
print("# Missing value 'month' variable: {0}".format(len(data1.loc[data1['month'] == "unknown"])))
print("# Missing value 'housing' variable: {0}".format(len(data1.loc[data1['housing'] == "unknown"])))
print("# Missing value 'balance' variable: {0}".format(len(data1.loc[data1['balance'] == "unknown"])))
print("# Missing value 'previous' variable: {0}".format(len(data1.loc[data1['previous'] == "unknown"])))
print("# Missing value 'loan' variable: {0}".format(len(data1.loc[data1['loan'] == "unknown"])))
print("# Missing value 'y' variable: {0}".format(len(data1.loc[data1['y'] == "unknown"])))
print("# Missing value 'poutcome' variable: {0}".format(len(data1.loc[data1['poutcome'] == "unknown"])))


# In[38]:


#For variable 'poutcome'- combine 'other' and 'unknown'
data1.poutcome.value_counts() #Check initial 'poutcome' entries

data1['poutcome'] = data1['poutcome'].replace(['other'],'unknown')

#Check 'poutcome' entries after combining 'other' and 'unknown'.
data1.poutcome.value_counts() 


# In[39]:


# Change 'month' from words to numbers for easier analysis
lst = [data1]
for column in lst:
    column.loc[column["month"] == "jan", "month_int"] = 1
    column.loc[column["month"] == "feb", "month_int"] = 2
    column.loc[column["month"] == "mar", "month_int"] = 3
    column.loc[column["month"] == "apr", "month_int"] = 4
    column.loc[column["month"] == "may", "month_int"] = 5
    column.loc[column["month"] == "jun", "month_int"] = 6
    column.loc[column["month"] == "jul", "month_int"] = 7
    column.loc[column["month"] == "aug", "month_int"] = 8
    column.loc[column["month"] == "sep", "month_int"] = 9
    column.loc[column["month"] == "oct", "month_int"] = 10
    column.loc[column["month"] == "nov", "month_int"] = 11
    column.loc[column["month"] == "dec", "month_int"] = 12
    
    
 #[code adapted from: https://stackoverflow.com/questions/3418050/month-name-to-month-number-and-vice-versa-in-python]


# In[40]:


print('The Dataframes shape is:', data1.shape)


# In[41]:


## job and education also have unknown values
# data.education.value_counts() 
# data.job.value_counts() 

no = data.loc[data['y'] == 'no']
yes = data.loc[data['y'] == 'yes']
unknown_no = data1.loc[((data1['job'] == 'unknown')|(data1['education'] == 'unknown'))&(data1['y'] == 'no')]
unknown_yes = data1.loc[((data1['job'] == 'unknown')|(data1['education'] == 'unknown'))&(data1['y'] == 'yes')]


print('The percentage of unknown values in class no: ', float(unknown_no.count()[0]/float(no.count()[0]))*100)
print('The percentage of unknown values in class yes: ', float(unknown_yes.count()[0]/float(yes.count()[0]))*100)


#[code adapted from:https://stackoverflow.com/questions/70327099/python-can-i-replace-missing-values-marked-as-e-g-unknown-to-nan-in-a-datafra]


# In[42]:


data.drop(no, axis=1)
data.drop(yes, axis=1)


# In[43]:


len(data1.loc[data1['contact'] == "unknown"])/len(data1.contact)


# In[44]:


data1['contact'] = data1['contact'].replace(['contact.mode'],'unknown')
print("# Missing value 'contact' variable: {0}".format(len(data1.loc[data1['contact'] == "unknown"])))


# In[45]:


data1['contact'] = data1['contact'].replace(['unknown'],'cellular')
data1.contact.value_counts() 


# In[46]:


data1['contact'] = data1['contact'].replace(['contact.mode'],'unknown')
print("# Missing value 'contact' variable: {0}".format(len(data1.loc[data1['contact'] == "unknown"])))


# In[47]:


data1.head()


# In[48]:


print('The Dataframes shape is:', data1.shape)


# In[49]:


print("Customers whom have not been previously contacted:", len(data1[data1.pdays==-1]))
print("Maximum values for the attribute pdays:", data1['pdays'].max())
data1.loc[data1['pdays'] == -1, 'pdays'] = 100000


# In[50]:


#Checking the presence of outliers using Box Plots

#Box plot visualization representing the relationship between the different variables with the target variable

fig,axarr = plt.subplots(2,3, figsize=(20,15), dpi=300, facecolor='w', edgecolor='k')
sns.set(style="white")
sns.boxplot(x='age',data = data1, ax=axarr[0][0], palette="winter")
axarr[0][0].set_title('Distribution of Age')
sns.boxplot(x='balance',data = data1, ax=axarr[0][1], palette="winter")
axarr[0][1].set_title('Distribution of balance')
sns.boxplot(x='campaign', data = data1,ax=axarr[0][2], palette="winter")
axarr[0][2].set_title('Distribution of campaign')
sns.boxplot(x='pdays',data = data1, ax=axarr[1][0], palette="winter")
axarr[1][0].set_title('Distribution of pdays')
sns.boxplot(x='previous', data = data1, ax=axarr[1][1], palette="winter")
axarr[1][1].set_title('Distribution of previous')

sns.boxplot(x='day', data = data1, ax=axarr[1][1], palette="winter")
axarr[1][1].set_title('Distribution of day')

fig.suptitle('Box plot representing relationship between Numeric Feature Variables and the Target Variable', fontsize=18);
plt.show()

#[code adapted from:https://datatofish.com/bar-chart-python-matplotlib/]


# In[ ]:





# In[51]:


data1.hist(column=['age', 'balance','campaign', 'pdays', 'previous', 'day', 'month_int'],figsize= (15,15), grid="grid")


# In[52]:


#Implement RobustScaler
Rscaler = RobustScaler()
num_cols = ['age', 'balance', 'campaign', 'pdays', 'previous', 'day', 'month_int']
data1[num_cols] = Rscaler.fit_transform(data1[num_cols])
data1.head()


# In[53]:


# To normalise the data use StandardScaler
Sscaler = StandardScaler()
num_cols = ['age', 'balance', 'campaign', 'pdays', 'previous', 'day', 'month_int']
data1[num_cols] = Sscaler.fit_transform(data1[num_cols])
data1.head()


# In[54]:


# Encoding catergorical data to numeric with LabelEncoder

categorical_to_numeric = preprocessing.LabelEncoder()
CatCols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome','y', 'contact']
for i in CatCols:
    data1[i] = categorical_to_numeric.fit_transform(data[i].values)
#[code adapted from:https://www.mygreatlearning.com/blog/label-encoding-in-python/]


# In[55]:


data1.head()


# In[56]:


# End of data preprocessing


# In[57]:


#(b) Top four classification feature selection


# In[ ]:





# In[59]:


import matplotlib.pyplot as pyplot


# configure to select all features
fs = SelectKBest(score_func=f_classif, k='all')
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)


# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

#[code adapted from:https://www.datatechnotes.com/2021/02/seleckbest-feature-selection-example-in-python.html]


# In[60]:


#Develop Correlation Matrix & Plot Heatmap
corr= data1.corr()
plt.figure(figsize = (12,12))
cmap = sns.diverging_palette(240, 10, n=9,as_cmap=True)
sns.heatmap(corr[corr>0.1], xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap=cmap,center=0,
            vmax=.3, linewidths=.4, square=True,
            cbar_kws={"shrink": .82},annot=True)

plt.title('Correlation Matrix Heatmap')
plt.show()


# In[61]:


corr.head()


# In[62]:


#Break down the features by the Class


sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "age").add_legend()
plt.show()


sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "marital").add_legend()
plt.show()

sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "default").add_legend()
plt.show()

sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "balance").add_legend()
plt.show()

sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "loan").add_legend()
plt.show()


sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "campaign").add_legend()
plt.show()

sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "housing").add_legend()
plt.show()

sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "pdays").add_legend()
plt.show()

sns.FacetGrid(data1, hue="y", height=5).map(sns.distplot, "contact").add_legend()
plt.show()

sns.FacetGrid(data1,hue="y", height=5).map(sns.distplot, "previous").add_legend()
plt.show()

#[code adapted from: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html]


# In[63]:


# To observe the feature variable's correlation with 'y', extract the y_cat column which represents the target variable


corr_y =pd.DataFrame(corr['y'].drop('y'))
corr_final=abs(corr_y.sort_values(by='y', ascending = True))
print(corr_final.nlargest(4,'y').head(4))


# In[64]:


#Class Imbalance Issue Visualized 
#Histogram of '1' and '0' in 'y'
data1.hist(column=['y'])
plt.title("Historgraph of 1 and 0 in 'y'")
plt.show()


# In[65]:


data1.y.value_counts()


# In[66]:


# (c) Building a decision tree and adjusting only two features

#Set Features (based on top 5 most inflential feature from previous step)
X = data1[['pdays','previous','housing','contact']]

#Set Target
y = data1['y']

#Prepare Training and Testing Data (20% test data)
X_train,X_test,y_train,y_test= train_test_split(X,y, shuffle=True, test_size=0.2, random_state=50)

#Display Training and Testing Data
print('Shape of the training feature:', X_train.shape)
print('Shape of the testing feature:', X_test.shape)
print('Shape of the training label:', y_train.shape)
print('Shape of the training label:', y_test.shape)


# In[67]:


from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
columns = X_train.columns
data_X, data_y = os.fit_resample(X_train, y_train)
smoted_X = pd.DataFrame(data=data_X,columns=columns )
smoted_y = pd.DataFrame(data=data_y,columns=['y'])


# In[68]:


X_train = smoted_X
y_train = smoted_y
X_test
y_test


# In[69]:


sns.countplot(x='y', data=smoted_y)
plt.show()


# In[70]:


#Construct the decision Tree Model


decision_tree = tree.DecisionTreeClassifier(random_state=42)

#Train the Decision Tree Classifier
train_dt=decision_tree.fit(smoted_X,smoted_y)

#Plot the Initial Decision Tree
plt.figure(dpi=400)
tree.plot_tree(train_dt)
plt.show()

print("number of nodes:", decision_tree.tree_.node_count)


# In[71]:


# Model Accuracy, how often is the classifier correct?
y_pred = decision_tree.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[73]:


#10 fold cross-validation score:
cross_val= cross_val_score(decision_tree,X,y,cv=10)
print(cross_val)
print("Averaged 10-Fold CV Score:{}".format(np.mean(cross_val)))


# In[74]:


#Plot Confusion Matrix with no changes in parameters
plot_confusion_matrix(train_dt, X_test, y_test, normalize= 'all')
plt.title('Confusion matrix of the classifier')
plt.show()


# In[75]:


# classfication report
svc = svm.SVC(kernel='rbf', C=70, gamma=0.001).fit(smoted_X,smoted_y)
predictionsvm = svc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictionsvm))


# In[76]:


#Tune 'max_depth' parameter.
maxdepth_cv=[]
node_counts=[]

for k in range(1,6,1):
     dt=DecisionTreeClassifier(max_depth=k,random_state=42)
     dt.fit(smoted_X,smoted_y)
     predict=dt.predict(X_test)
     cv= cross_val_score(dt,X,y,cv=10)
     nodecount = dt.tree_.node_count
     print("max_depth={}".format(k),
           "Average 10-Fold CV Score:{}".format(np.mean(cv)),
           "Node count:{}".format(nodecount))
     maxdepth_cv.append(np.mean(cv))
     node_counts.append(nodecount)
    
#[code adpated from: https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/]


# In[77]:


#Plot averaged CV scores for all max_depth tunings
fig,axes=plt.subplots(1,1,figsize=(8,5))
axes.set_xticks(range(1,6,1))
k=range(1,6,1)
plt.plot(k,maxdepth_cv)
plt.xlabel("max_depth")
plt.ylabel("Averaged 10-fold CV score")
plt.show()


# In[78]:


#Plot Decision Tree with (max_depth=2)
dt_depth2 = tree.DecisionTreeClassifier(max_depth=2,random_state=42)
tdt_depth2=dt_depth2.fit(smoted_X,smoted_y)

plt.figure(dpi=300)
tree.plot_tree(tdt_depth2, feature_names=X.columns, class_names='Subscribed' 'Not Subscribed')
plt.title("Decision Tree Diagram (max_depth = 2)")

plt.show()


# In[79]:



#Plot Confusion Matrix for depth of 2
plot_confusion_matrix(tdt_depth2, X_test, y_test, normalize= 'all')
plt.title('Confusion matrix of the classifier')
plt.show()


# In[80]:


# classfication report when max_depth parameter=2 
svc = svm.SVC(kernel='rbf', C=70, gamma=0.001).fit(smoted_X,smoted_y)
predictionsvm = svc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictionsvm))


# In[81]:


# classfication report of the original decision tree
svc = svm.SVC(kernel='rbf', C=70, gamma=0.001).fit(X_test,y_test)
predictionsvm = svc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictionsvm))


# In[ ]:


#Tune 'max_leaf_nodes' parameter.
maxleaf_cv=[]
node_counts=[]


for k in range(2,11,1): 
     dt=DecisionTreeClassifier(max_leaf_nodes=k,random_state=42)
     dt.fit(smoted_X,smoted_y)
     predict=dt.predict(X_test)
     cv= cross_val_score(dt,X,y,cv=10)
     nodecount = dt.tree_.node_count
     print("max_leaf_nodes count={}".format(k),
           "Average 10-Fold CV Score:{}".format(np.mean(cv)),
           "Node counts:{}".format(nodecount))
     maxleaf_cv.append(np.mean(cv))
     node_counts.append(nodecount)
    
    
#[code adpated from: https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/]


# In[ ]:


#Plot averaged CV scores for all tuned max_leaf_nodes tunings
fig,axes=plt.subplots(1,1,figsize=(8,5))
axes.set_xticks(range(2,11,1))
k=range(2,11,1)
plt.plot(k,maxleaf_cv)
plt.xlabel("max_leaf_nodes count")
plt.ylabel("Averaged 10-fold CV score")
plt.show()


# In[ ]:


#Plot Decision Tree with (max_leaf_nodes=10)
dt_leaf10 = tree.DecisionTreeClassifier(max_leaf_nodes=10,random_state=50)
tdt_leaf10=dt_leaf10.fit(smoted_X,smoted_y)

plt.figure(dpi=300)
tree.plot_tree(tdt_leaf10,  feature_names=X.columns, class_names='Subscribed' 'Not Subscribed')
plt.title("Decision Tree Diagram (max_leaf_nodes = 10)")
plt.show()

#[code adpated from: https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/]


# In[ ]:



#Plot Confusion Matrix
plot_confusion_matrix(dt_leaf10, X_test, y_test, normalize= 'all')
plt.title('Confusion matrix of the classifier')
plt.show()


# In[ ]:


#Decision Tree with (max_leaf_nodes=5)
dt_leaf5 = tree.DecisionTreeClassifier(max_leaf_nodes=5,random_state=42)
tdt_leaf5=dt_leaf5.fit(smoted_X,smoted_y)
plt.figure(dpi=300)
tree.plot_tree(tdt_leaf5, feature_names=X.columns, class_names='Subscribed' 'Not Subscribed')
plt.title("Decision Tree Diagram (max_leaf_nodes = 5)")
plt.show()


# In[ ]:


svc = svm.SVC(kernel='rbf', C=70, gamma=0.001).fit(smoted_X,smoted_y)
predictionsvm = svc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictionsvm))


# In[ ]:


predictionsvm = svc.predict(X_test)
percentage = svc.score(X_test,y_test)
percentage


# In[ ]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


cm = confusion_matrix(y_test, predictionsvm, labels=[0, 1])
print(cm)


# In[ ]:



#Plot Confusion Matrix
plot_confusion_matrix(dt_leaf5, X_test, y_test, normalize= 'all')
plt.title('Confusion matrix of the classifier')
plt.show()


# In[ ]:



# QUESTION TWO


# In[ ]:





# In[82]:


# Initial Dataset Examination
#import packages
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from scipy.io import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[83]:


#Import the file
data = arff.loadarff("Autism-Child-Data.arff")

#Converting the file to a DataFrame
autism_df= pd.DataFrame(data[0])
autism_df.head()


# In[84]:


# data pre-processing


# In[86]:


#Change the encodings of the character to rid off the b's
def decode(df_name):
    for col in df_name.columns:
        if df_name[col].dtype != 'float64':
            df_name[col] = df_name[col].apply(lambda val : val.decode('utf-8'))
    pd.set_option('display.max_columns',50)
    return df_name

autism = decode(autism_df)


# In[87]:


autism.head()


# In[88]:


autism.info()


# In[89]:


autism.shape


# In[90]:


sns.countplot(x='Class/ASD', data=autism)
plt.show()


# In[92]:


autism.columns


# In[95]:


# Fixing the typos in the column name
#'austim' to 'autism'
autism = autism.rename(columns = {'austim':'autism'})

#'jundice'  to 'jaundice'
autism = autism.rename(columns = {'jundice':'jaundice'})

#'contry_of_res' to 'country_of_res'
autism = autism.rename(columns = {'contry_of_res':'country_of_res'})


# In[96]:


# Drop Columns #
#Drop 'relation' column because of low relevance for prediction
autism.drop('relation',axis=1, inplace=True)

#Drop 'used_app_before' columnbecause of low relevance for prediction
autism.drop('used_app_before',axis=1, inplace=True)

#Drop 'age_desc' column since it only contains a single value describing the age range of child subjects
autism.drop('age_desc',axis=1, inplace=True)

#Drop 'result' column - as its a congregate colum of A1- A10 scores.
autism.drop('result',axis=1, inplace=True)

#Check columns after dropping
autism.columns


# In[97]:


#Bar plot visualization of feature variables' relationship with target variable
fig,axarr = plt.subplots(2,3, figsize=(17,10), dpi=300, facecolor='w', edgecolor='k')
sns.set(style="white")
sns.countplot(x='age', hue = 'Class/ASD',data = autism, ax=axarr[0][0], palette="coolwarm")
axarr[0][0].set_title('Distribution of Age')
sns.countplot(x='gender', hue = 'Class/ASD',data = autism, ax=axarr[0][1], palette="coolwarm")
axarr[0][1].set_title('Distribution of Gender')
sns.countplot(x='ethnicity', hue = 'Class/ASD',data = autism,ax=axarr[0][2], palette="coolwarm")
axarr[0][2].set_title('Distribution of ethnicity')
sns.countplot(x='jaundice', hue = 'Class/ASD',data = autism, ax=axarr[1][0], palette="coolwarm")
axarr[1][0].set_title('Distribution of jaundice')
sns.countplot(x='autism', hue = 'Class/ASD',data = autism, ax=axarr[1][1], palette="coolwarm")
axarr[1][1].set_title('Distribution of autism')
sns.countplot(x='country_of_res', hue = 'Class/ASD',data = autism, ax=axarr[1][2], palette="coolwarm")
axarr[1][2].set_title('Distribution of country_of_res')


fig.suptitle('Distribution of the Feature Variables against Target Variable', fontsize=16);
plt.show()

#[code adapted from:https://seaborn.pydata.org/generated/seaborn.countplot.html]


# In[99]:


#Exploring null values
autism.isnull().sum()


# In[100]:



#Exporing 'Age' containing with null
autism[autism['age'].isnull()]


# In[101]:


#Replace the records found in 'Age' with null values with '0'
autism['age']=autism['age'].fillna(value=0)
autism[autism['age'] == 0] #Check


# In[102]:


# Verifying the step of replacing above
autism.isnull().sum()


# In[103]:


#Converting 'Age' column's datatype from FLOAT to INT.
autism['age']=autism['age'].astype('int')

#Verifying the conversion
autism['age'].dtype


# In[104]:


#Convert all 'A()_Score' columns  to INT.
def scores(df_name, cols_lst):
    for col in cols_lst:
        df_name[col] = df_name[col].astype('int')
    return df_name

scores(autism,['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score'])
autism['A10_Score'].dtype #check


# In[105]:


autism['A1_Score'].dtype


# In[106]:


#Convert 'gender' column - Change 'm'(male) to '1', 'f'(female) to '0'
autism['gender'].value_counts() #first check 'gender' values
autism['gender'] = autism['gender'].map({'m':1,'f':0})
autism['gender'].value_counts() #Check 'gender' values after replacement


# In[107]:


#Convert 'jaundice' column - Change 'yes' to '1', 'no' to '0'
autism['jaundice'].value_counts() #first check 'jaundice' values
autism['jaundice'] = autism['jaundice'].map({'yes':1,'no':0})
autism['jaundice'].value_counts() #Check 'jaundice' values after replacement


# In[108]:


#Convert 'Class/ASD' column - Change 'YES' to '1', 'NO' to '0'
autism['Class/ASD'].value_counts() #first check 'Class/ASD' values
autism['Class/ASD'] = autism['Class/ASD'].map({'YES':1,'NO':0})
autism['Class/ASD'].value_counts() #Check 'Class/ASD' values after replacement


# In[109]:


#Convert 'autism' column - Change 'yes' to '1', 'no' to '0'
autism['autism'].value_counts() #first check 'autism' values
autism['autism'] = autism['autism'].map({'yes':1,'no':0})
autism['autism'].value_counts() #Check 'autism' values after replacement


# In[110]:


#Check data types to make sure all necessary conversions has been done.
autism.info(verbose=True)


# In[111]:



#Find Nulls recorded as '?'
autism.shape
autism.isin(['?']).sum()


# In[ ]:


# 43 records in 'ethnicity' identified not identified correctly so would be replaced by unknown
autism['ethnicity'].replace({"?": "Unknown"}, inplace=True)


# In[112]:


# Encoding catergorical data to numeric with LabelEncoder
le = preprocessing.LabelEncoder()
CatCols = ['ethnicity', 'country_of_res']
for i in CatCols:
    autism[i] = le.fit_transform(autism[i].values)


# In[113]:


autism.head()


# In[114]:


# Top 5 feature selection


# In[115]:


#split dataset
X= autism.drop(['Class/ASD'], axis = 1) #feature variables
y= autism['Class/ASD']    #target variable


# In[116]:


#Using SelectKBest to identify the top 5 best features
KBest = SelectKBest(score_func=chi2, k=5)
fit = KBest.fit(X,y)
scores = pd.DataFrame(fit.scores_)
columns = pd.DataFrame(X.columns)

#Join results for visualization
featureScores = pd.concat([columns,scores],axis=1)
featureScores.columns = ['Feature','Score']
print(featureScores.nlargest(5,'Score').head(5))

#[code adapted from: https://www.datatechnotes.com/2021/02/seleckbest-feature-selection-example-in-python.html]


# In[119]:


# Verfification using the Correlation Matrix & Heatmap
autism_corr= autism.corr()
plt.figure(figsize = (20,15))
cmap = sns.diverging_palette(240, 10, n=9,as_cmap=True)
sns.heatmap(autism_corr[autism_corr>0.4],annot=True)
plt.title('ASD Pearson Correlation Heatmap')
plt.show()


# In[120]:


#Observing the correlations with 'Class/ASD' in descending order
autism_corr_y =pd.DataFrame(autism_corr['Class/ASD'].drop('Class/ASD'))
autism_corr_final=abs(autism_corr_y.sort_values(by='Class/ASD', ascending = False))
print(autism_corr_final.nlargest(5,'Class/ASD').head(5))


# In[121]:


# Plotting the breakdown analysis for the selected features
fig,axarr = plt.subplots(2,3, figsize=(17,10), dpi=300, facecolor='w', edgecolor='k')
sns.set(style="white")



sns.countplot(x='A4_Score', hue = 'Class/ASD',data = autism, ax=axarr[0][0], palette="coolwarm")
axarr[0][0].set_title('Distribution of A4_Score')


sns.countplot(x='A9_Score', hue = 'Class/ASD',data = autism, ax=axarr[0][1], palette="coolwarm")
axarr[0][1].set_title('Distribution of A9_Score')



sns.countplot(x='A10_Score', hue = 'Class/ASD',data = autism,ax=axarr[0][2], palette="coolwarm")
axarr[0][2].set_title('Distribution of A10_Score')



sns.countplot(x='A8_Score', hue = 'Class/ASD',data = autism, ax=axarr[1][0], palette="coolwarm")
axarr[1][0].set_title('Distribution of A8_Score')



sns.countplot(x='A6_Score', hue = 'Class/ASD',data = autism, ax=axarr[1][1], palette="coolwarm")
axarr[1][1].set_title('Distribution of A6_Score')





fig.suptitle('Distribution of Feature Variables vs Target Variable', fontsize=16);
plt.show()



# In[122]:


#Break down the features by the Class


sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A1_Score").add_legend()
ax=axarr[0][0]
plt.show()


sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A2_Score").add_legend()
ax=axarr[0][1]
plt.show()


sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A3_Score").add_legend()
plt.show()


sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A4_Score").add_legend()
plt.show()

sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A5_Score").add_legend()
plt.show()

sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A6_Score").add_legend()
plt.show()

sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A7_Score").add_legend()
plt.show()
sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A8_Score").add_legend()
plt.show()

sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A9_Score").add_legend()
plt.show()

sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "A10_Score").add_legend()
plt.show()


sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "age").add_legend()
plt.show()


sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "gender").add_legend()
plt.show()

sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "ethnicity").add_legend()
plt.show()
sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "jaundice").add_legend()
plt.show()
sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "autism").add_legend()
plt.show()

sns.FacetGrid(autism, hue="Class/ASD", height=5).map(sns.distplot, "country_of_res").add_legend()
plt.show()

#[code adapated from: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html]


# In[ ]:


# libraries & dataset
import seaborn as sns
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")

 
# Grouped violinplot
sns.violinplot(x="gender",y="Class/ASD" , hue="Class/ASD", data=autism, palette="Pastel1")
plt.show()
sns.violinplot(x="country_of_res",  hue="Class/ASD", data=autism, palette="Pastel1")
plt.show()


# In[ ]:


# Running the Naive Bayes algorithm with GaussianNB implenetation for the selected features


# In[ ]:


#Create Training and Testing data
#Set Features (based on best 5 correlations from previous step)
X= autism[['A4_Score','A9_Score','A8_Score','A1_Score','A10_Score']]

#Set Target
y= autism['Class/ASD']

#Prepare Training and Testing Data (20% test data)
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)

#No need to apply scaler since data are all 1 and 0

#Display Training and Testing Data
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)


# In[ ]:


X.head()


# In[ ]:


#Build GaussianNB Model
gnb = GaussianNB() 
gnb.fit(X_train, np.ravel(y_train,order='C')) 
predictions = gnb.predict(X_test)


# In[ ]:


#Computer 10-fold Cross-Validation Score
cv = cross_val_score(gnb,X_test,y_test, cv=10)
print("Average 10-Fold CV Score - GaussianNB:{}".format(np.mean(cv)))


# In[ ]:


#Compute Accuracy Score
from sklearn.metrics import accuracy_score
print("Accuracy score of Gaussian Naive Bayes Model:", accuracy_score(y_test, predictions))


# In[ ]:


confusion_matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
print(confusion_matrix)


# In[ ]:


# confusion matrix of Naive bayes algorithm model
plot_confusion_matrix(gnb,X_test,y_test, normalize='all')
plt.show()


# In[ ]:


#Classification Report
report = classification_report(y_test, predictions, target_names=['0', '1'])
print(report)


# In[ ]:


#Run Decision Tree Classifer
#Use original data before feature selection
autism.head()

#Set Features
Xtree= autism.drop(['Class/ASD'],axis=1) 
#Set Target
ytree= autism['Class/ASD']

#Prepare Training and Testing Data (20% test data)
X_train,X_test,y_train,y_test= train_test_split(Xtree,ytree, test_size=0.2, random_state=42)


# In[ ]:


#Create and fit Decision Tree Classifier to data before feature selection
tree= tree.DecisionTreeClassifier(random_state=42)
tree = tree.fit(X_train, y_train)
predictions= tree.predict(X_test)

plt.figure(dpi=400)
plt.show()


# In[ ]:


#Decision Tree with (max_leaf_nodes=5)
tree = tree.DecisionTreeClassifier(random_state=42)
tree=tree.fit(X_train, y_train)
plt.figure(dpi=300)
predictions= tree.predict(X_test


tree.plot_tree(tree, feature_names=X.columns, class_names='Yes' 'No')
plt.title("Decision Tree Diagram (max_leaf_nodes = 5)")
plt.show()


# In[123]:


#Plotting the Decision Tree Feature Importance
Feature_Importance=pd.Series(tree.feature_importances_,index=Xtree.columns)
Feature_Importance.sort_values(ascending=False, inplace=True)
Feature_Importance.plot.bar()
plt.title("Decision Tree Feature")
plt.show()


# In[ ]:


# confusion matrix of decision tree
plot_confusion_matrix(tree,X_test,y_test, normalize='all')
plt.show()


# In[ ]:


#Computing the 10-fold Cross-Validation Score
cv = cross_val_score(tree,X_test,y_test, cv=10)
print("Average 10-Fold CV Score - Decision Tree:{}".format(np.mean(cv)))


# In[ ]:


#Compute Accuracy Score
from sklearn.metrics import accuracy_score
print("Accuracy score of the Decision Tree is :", accuracy_score(y_test, predictions))


# In[ ]:


#Classification Report for the decision tree
report = classification_report(y_test, predictions, target_names=['0', '1'])
print(report)


# In[ ]:




