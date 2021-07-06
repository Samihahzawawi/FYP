#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install statsmodels')


# In[2]:


#import all the libraries that needed

get_ipython().system('pip install pandas')
import pandas as pd #to work with data
import numpy as np #to work with arrays
get_ipython().system('pip install seaborn')
import seaborn as sns #for visualization
import matplotlib.pyplot as plt #for visualization
get_ipython().system('pip install statsmodels')
import statsmodels.formula.api as sn #for classes
import scipy.stats as stats #for skew the identification features

from matplotlib.backends.backend_pdf import PdfPages #save plot to pdf
get_ipython().system('pip install scikit-learn')
from sklearn.model_selection import train_test_split #split malicious dataset from training and testing 
from sklearn import metrics #for the accuracy score
from sklearn.linear_model import LogisticRegression #to predict categorical dependent var
get_ipython().run_line_magic('conda', 'install statsmodels')
get_ipython().system('pip install statsmodels')
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings #to ignore warnings
warnings.filterwarnings("ignore")

#dataset settings
pd.set_option('display.max_columns', None)
#np.set_printoptions(threshold = np.nan)
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision = 3)

sns.set(style = "darkgrid")

plt.rcParams ["axes.labelsize"] = 14
plt.rcParams ["xtick.labelsize"] = 12
plt.rcParams ["ytick.labelsize"] = 12


# In[3]:


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "last_flag"]


# In[4]:


#import train set
import pandas as pd
df = pd.read_csv(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTrain+.txt')
df = pd.read_table(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTrain+.txt', sep = ',', names = col_names)
df = df.iloc[:, :-1] #remove unwanted field name


# In[5]:


#import test set
df_test = pd.read_csv(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTest+.txt')
df_test = pd.read_table(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTest+.txt', sep = ',', names = col_names)
df_test = df_test.iloc[:, :-1] #remove unwanted field names


# In[6]:


df.info() #to get the info of dataset whether it have numerical or non-numerical values of features


# In[7]:


df.columns #attribute contains column names
print(df.columns)
df.index #attribute row nums or row names


# In[8]:


#to count the number of normal and attacks data in train and test set
print('The training set dimensions is:', df.shape) 
print('The test set dimensions is:', df_test.shape) 


# In[9]:


#train dataset sample view
#first 5 rows train dataset
#col_names = ['duration', 'protocol_type', 'service', 'flag', 'source_bytes', 'destination_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'number_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_count', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'last_flag']
#df = pd.read_table(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTrain+.txt', sep = ',', names = col_names)
#df = pd.read_csv(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTrain+.txt')
df.head(5) #to returns first few rows of dataframe


# In[10]:


#first 5 rows test dataset
df_test.head(5) #to returns first few rows of dataframe


# In[11]:


#a statistic summary
#col_names = ['duration', 'protocol_type', 'service', 'flag', 'source_bytes', 'destination_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'number_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_count', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'last_flag']
#df = pd.read_csv(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTest+.txt')
#df_test = pd.read_table(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTest+.txt', sep = ',', names = col_names)
#df = pd.read_csv(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTrain+.txt')
#df = pd.read_table(r'C:\Users\miazawawi\Desktop\FYPCoding\NSL_KDD-master\KDDTrain+.txt', sep = ',', names = col_names)
df.describe() #to describe method that computes statistics for numerical columns like mean, median 
#count = num of non missing values in each columns


# In[12]:


#to view the details numbers of normal and specific number of attacks
#To get the train set label distribution
print('Label distribution of train set:')
print(df['label'].value_counts())
print()


# In[13]:


#same goes to test set
print('Label distribution of test set:')
print(df_test['label'].value_counts())


# In[14]:


#Pre-processing Feature
#Using One-Hot-Encoding to make all features numerical
#1.Identify categorical features in the dataset.
#2. Columns protocol_type, service, flag

print("Training set:")
for col_name in df.columns:
    if df[col_name].dtypes == "object" :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name = col_name, unique_cat=unique_cat))
print()
#to just zoom in/to see only one column (service) only
print("Distribution of categories in service:")
#to sort row passing columns name that want to sort by, ascending = False is sort from weight to lighter
print(df["service"].value_counts().sort_values(ascending=False).head()) 


# In[15]:


#to find categorical features, same loop applied on test set
print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len (df_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


# In[16]:


#to get in variable and categorical columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
categorical_columns=["protocol_type", "service", "flag"]
df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
df_categorical_values.head()


# In[17]:


#column names are created for dummies
#column names for protocol_type + protocol rows that categorical not binary yet
#protocol has 3 dummies
unique_protocol = sorted(df.protocol_type.unique())
string1 = "Protocol_type_"
unique_protocol2= [string1 + x for x in unique_protocol]

#service has 64 dummies
#column names for service + service rows
unique_service = sorted(df.service.unique())
string2 = "service_"
unique_service2= [string2 + x for x in unique_service]

#flag has 11 dummies
#column names for flag + flag rows
unique_flag = sorted(df.flag.unique())
string3 = "flag_"
unique_flag2= [string3 + x for x in unique_flag]

#put all the dummies together
dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)


# In[18]:


#column names dummies for test also created
unique_service_test = sorted(df_test.service.unique())
unique_service2_test = [string2 + x for x in unique_service_test]

##put all the test dummies together
testdumcols = unique_protocol2 + unique_service2_test + unique_flag2
print(dumcols)


# In[19]:


#LabelEncoder() are used to transform categorical/object features into numbers
#convert each value in a column to a number
#for train set

df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)
print(df_categorical_values_enc.head())

#do the same step
#for test set

testdf_categorical_values_enc = testdf_categorical_values.apply(LabelEncoder().fit_transform)
print(testdf_categorical_values_enc.head())


# In[20]:


#to convert categorical data to numerical data using One-Hot Encoding since LabelEncoder can get misinterpreted as have some sort of hierarchy
#convert each value in a column to a number
#create instance of one-hot-encoder

#for train set
enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform (df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(), columns = dumcols)

#for test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(), columns = testdumcols)

df_cat_data.head()


# In[21]:


#6 missing categories in service are added from train set to test set.
trainservice = df["service"].tolist()
testservice = df_test["service"].tolist()
difference = list(set(trainservice) - set(testservice))
string = "service_"
difference = [string + x for x in difference]
difference


# In[22]:


#recheck row and columns of test set
for col in difference:
    testdf_cat_data[col] = 0

testdf_cat_data.shape


# In[23]:


#categorical dataframe joined with non-categorical/numerical dataframe
#for train data

newdf = df.join(df_cat_data)
newdf.drop("flag", axis = 1, inplace = True)
newdf.drop("protocol_type", axis = 1, inplace = True)
newdf.drop("service", axis = 1, inplace = True)

#for test data
newdf_test = df_test.join(testdf_cat_data)
newdf_test.drop("flag", axis = 1, inplace = True)
newdf_test.drop("protocol_type", axis = 1, inplace = True)
newdf_test.drop("service", axis = 1, inplace = True)
print(newdf.shape)
print(newdf_test.shape)


# In[24]:


#Dataset are split into 4 datasets since theres are 4 category attacks:
#Rename each attacks with:
#Attack Class = 0:Normal, 1:Dos, 2:Probe, 3:R2L, 4:U2R
#0= Normal
#1=Dos: Back, Land, Neptune, Pod, Smurf, Teardrop, Apache2, Udpstorm, Processtable, Worm(10)
#2=Probe: Satan, Ipsweep, Nmap, Portsweep, Mscan, Saint(6)
#3=R2L: Guess_Password, Ftp_write,Imap, Phf, Multihop, Warezmaster, Warezclient,Spy, Xlock, Xsnoop, Snmguess, Snmpgetattack, Httptunnel, Sendmail, Named(16)
#4=U2R: Buffer_overflow, Loadmodule, Rootkit, Perl, Sqlattack, Xterm, Ps(7)

#take the "label" column from train set
labeldf=newdf["label"]
#change label column for train
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1,
                            'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2,
                           'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,
                            'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,
                            'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
#put back the new label column 
newdf["label"]=newlabeldf
print(newdf["label"].head())


# In[25]:


#repeat the same step in order to change the column "label"
labeldf_test=newdf_test["label"]
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 
                                      'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                                      'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2,
                                      'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,
                                      'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,
                                      'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                                      'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newdf_test["label"]=newlabeldf_test
print(newdf_test["label"].head())


# In[26]:


#train
to_drop_DoS=[2,3,4]
to_drop_Probe=[1,3,4]
to_drop_R2L=[1,2,4]
to_drop_U2R=[1,2,3]

DoS_df=newdf[~newdf["label"].isin(to_drop_DoS)];
Probe_df=newdf[~newdf["label"].isin(to_drop_Probe)];
R2L_df=newdf[~newdf["label"].isin(to_drop_R2L)];
U2R_df=newdf[~newdf["label"].isin(to_drop_U2R)];

print("Dos train Dimensions is:" ,DoS_df.shape)
print("Probe train Dimensions is:" ,Probe_df.shape)
print("R2L train Dimensions is:" ,R2L_df.shape)
print("U2R train  Dimensions is:" ,U2R_df.shape)


# In[27]:


#test
to_drop_DoS=[2,3,4]
to_drop_Probe=[1,3,4]
to_drop_R2L=[1,2,4]
to_drop_U2R=[1,2,3]

DoS_df_test=newdf_test[~newdf_test["label"].isin(to_drop_DoS)];
Probe_df_test=newdf_test[~newdf_test["label"].isin(to_drop_Probe)];
R2L_df_test=newdf_test[~newdf_test["label"].isin(to_drop_R2L)];
U2R_df_test=newdf_test[~newdf_test["label"].isin(to_drop_U2R)];

print("Dos test Dimensions is:" ,DoS_df_test.shape)
print("Probe test Dimensions is:" ,Probe_df_test.shape)
print("R2L test Dimensions is:" ,R2L_df_test.shape)
print("U2R test Dimensions is:" ,U2R_df_test.shape)


# In[28]:


#For Feature Scaling
#Split dataframes into X & Y
#assign X as a dataframe of features and Y as a series of outcome variables
X_DoS = DoS_df.drop('label',1)
Y_DoS = DoS_df.label
X_Probe = Probe_df.drop('label',1)
Y_Probe = Probe_df.label
X_R2L = R2L_df.drop('label',1)
Y_R2L = R2L_df.label
X_U2R = U2R_df.drop('label',1)
Y_U2R = U2R_df.label
# test set
X_DoS_test = DoS_df_test.drop('label',1)
Y_DoS_test = DoS_df_test.label
X_Probe_test = Probe_df_test.drop('label',1)
Y_Probe_test = Probe_df_test.label
X_R2L_test = R2L_df_test.drop('label',1)
Y_R2L_test = R2L_df_test.label
X_U2R_test = U2R_df_test.drop('label',1)
Y_U2R_test = U2R_df_test.label


# In[29]:


colNames=list(X_DoS)
colNames_test=list(X_DoS_test)


# In[30]:


from sklearn import preprocessing

#Train dataset 
scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS = scaler1.transform(X_DoS) 
scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe = scaler2.transform(X_Probe) 
scaler3 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L = scaler3.transform(X_R2L) 
scaler4 = preprocessing.StandardScaler().fit(X_U2R)
X_U2R = scaler4.transform(X_U2R) 

#Test dataset
scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test = scaler5.transform(X_DoS_test) 
scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test = scaler6.transform(X_Probe_test) 
scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test = scaler7.transform(X_R2L_test) 
scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test = scaler8.transform(X_U2R_test)


# In[31]:


print(X_DoS.std(axis=0))


# In[32]:


X_Probe.std(axis=0);
X_R2L.std(axis=0);
X_U2R.std(axis=0);


# In[80]:


from sklearn.feature_selection import SelectPercentile, f_classif
np.seterr(divide="ignore", invalid="ignore");
selector=SelectPercentile(f_classif, percentile=10)


# In[79]:


#DoS
#to know how many features were selected for DoS
X_newDoS=selector.fit_transform(X_DoS, Y_DoS)
X_newDoS.shape


# In[35]:


#from the 13 features that selected.
#to get the list of 13 features:
true=selector.get_support()
newcolindex_DoS=[i for i, x in enumerate(true) if x]
newcolname_DoS=list(colNames[i] for i in newcolindex_DoS)
newcolname_DoS


# In[36]:


#Probe
#to know how many features were selected for Probe
X_newProbe=selector.fit_transform(X_Probe,Y_Probe)
X_newProbe.shape


# In[37]:


#from the 13 features that selected.
#to get the list of 13 features:
true=selector.get_support()
newcolindex_Probe=[i for i, x in enumerate(true) if x]
newcolname_Probe=list(colNames[i] for i in newcolindex_Probe)
newcolname_Probe


# In[38]:


#R2L
#to know how many features were selected for R2L
X_newR2L=selector.fit_transform(X_R2L, Y_R2L)
X_newR2L.shape


# In[39]:


#from the 13 features that selected.
#to get the list of 13 features:
true=selector.get_support()
newcolindex_R2L=[i for i, x in enumerate(true) if x]
newcolname_R2L=list(colNames[i] for i in newcolindex_R2L)
newcolname_R2L


# In[40]:


#U2R
#to know how many features were selected for U2R
X_newU2R=selector.fit_transform(X_U2R, Y_U2R)
X_newU2R.shape


# In[41]:


#from the 13 features that selected.
#to get the list of 13 features:
true=selector.get_support()
newcolindex_U2R=[i for i, x in enumerate(true) if x]
newcolname_U2R=list(colNames[i] for i in newcolindex_U2R)
newcolname_U2R


# In[42]:


print("Dos with 13 selected features:", newcolname_DoS)
print()
print("Probe with 13 selected features:", newcolname_Probe)
print()
print("R2L with 13 selected features:", newcolname_R2L)
print()
print("U2R with 13 selected features:", newcolname_U2R)
print()


# In[43]:


#Another kind of feature selection by RFE
#To create a decision tree classifier

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

#Rank all 41 features and choose the best by sorted the features
rfe = RFE(clf, n_features_to_select=1)


# In[44]:


#RFE DoS
rfe.fit(X_newDoS, Y_DoS)
print ("DoS with 13 selected and sorted rank features:")
#using this code
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_DoS)))


# In[45]:


#RFE Probe
rfe.fit(X_newProbe, Y_Probe)
print ("Probe with 13 selected and sorted rank features:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_Probe)))


# In[46]:


#RFE R2L
rfe.fit(X_newR2L, Y_R2L) 
print ("R2L with 13 selected and sorted rank features:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_R2L)))


# In[47]:


#RFE U2R
rfe.fit(X_newU2R, Y_U2R) 
print ("U2R with 13 selected and sorted rank features:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_U2R)))


# In[48]:


#continue of RFE
from sklearn.feature_selection import RFE
clf = DecisionTreeClassifier(random_state=0)
rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
rfe.fit(X_DoS, Y_DoS)
X_rfeDoS=rfe.transform(X_DoS)
true=rfe.support_
rfecolindex_DoS=[i for i, x in enumerate(true) if x]
rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)


# In[49]:


rfe.fit(X_Probe, Y_Probe)
X_rfeProbe=rfe.transform(X_Probe)
true=rfe.support_
rfecolindex_Probe=[i for i, x in enumerate(true) if x]
rfecolname_Probe=list(colNames[i] for i in rfecolindex_Probe)


# In[50]:


rfe.fit(X_R2L, Y_R2L)
X_rfeR2L=rfe.transform(X_R2L)
true=rfe.support_
rfecolindex_R2L=[i for i, x in enumerate(true) if x]
rfecolname_R2L=list(colNames[i] for i in rfecolindex_R2L)


# In[51]:


rfe.fit(X_U2R, Y_U2R)
X_rfeU2R=rfe.transform(X_U2R)
true=rfe.support_
rfecolindex_U2R=[i for i, x in enumerate(true) if x]
rfecolname_U2R=list(colNames[i] for i in rfecolindex_U2R)


# In[52]:


print('Features selected for DoS:',rfecolname_DoS)
print()
print('Features selected for Probe:',rfecolname_Probe)
print()
print('Features selected for R2L:',rfecolname_R2L)
print()
print('Features selected for U2R:',rfecolname_U2R)


# In[53]:


#Size of each data frame
print(X_rfeDoS.shape)
print(X_rfeProbe.shape)
print(X_rfeR2L.shape)
print(X_rfeU2R.shape)


# In[72]:


#Step4: DecisionTree model are built
#All features and selected features are trained for later comparison (Prediction, cross validation)


#clf=DecisionTreeClassifier(random_state=0)
#All features
from sklearn.tree import DecisionTreeClassifier
clf_DoS=DecisionTreeClassifier()
clf_Probe=DecisionTreeClassifier()
clf_R2L=DecisionTreeClassifier()
clf_U2R=DecisionTreeClassifier()
clf_DoS.fit(X_DoS, Y_DoS)
clf_Probe.fit(X_Probe, Y_Probe)
clf_R2L.fit(X_R2L, Y_R2L)
clf_U2R.fit(X_U2R, Y_U2R)
clf_DoS.predict_proba([[2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]])
array([[0., 1.]])


# In[ ]:


#Selected features
#X_newU2R, newcolname_U2R, newcolindex_R2L
clf_rfeDoS=DecisionTreeClassifier(random_state=0)
clf_rfeProbe=DecisionTreeClassifier(random_state=0)
clf_rfeR2L=DecisionTreeClassifier(random_state=0)
clf_rfeU2R=DecisionTreeClassifier(random_state=0)
clf_rfeDoS.fit(X_rfeDoS, Y_DoS)
clf_rfeProbe.fit(X_rfeProbe, Y_Probe)
clf_rfeR2L.fit(X_rfeR2L, Y_R2L)
clf_rfeU2R.fit(X_rfeU2R, Y_U2R)


# In[ ]:


#Results
#Confusion matrices that included 41 features for each of the column/category
#0:normal, 1:attack
#to observe predicted probalities
clf_DoS.predict(X_DoS_test)[0:5]


# In[ ]:


#to observe preidcted probalities
clf_DoS.predict_proba(X_DoS_test)[0:5]


# In[ ]:


#Dos
Y_DoS_pred = clf_DoS.predict(X_DoS_test)
pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames = ["Actual:"], colnames = ["Prediction of DoS:"])


# In[ ]:


#Probe
Y_Probe_pred = clf_Probe.predict(X_Probe_test)
pd.crosstab(Y_Probe_test, Y_Probe_pred, rownames = ["Actual:"], colnames=["Prediction of Probe:"])


# In[ ]:


#R2L
Y_R2L_pred = clf_R2L.predict(X_R2L_test)
pd.crosstab(Y_R2L_test, Y_R2L_pred, rownames = ["Actual:"], colnames = ["Prediction of R2L:"])


# In[ ]:


#U2R
Y_U2R_pred = clf_U2R.predict(X_U2R_test)
pd.crosstab(Y_U2R_test, Y_U2R_pred, rownames = ["Actual:"], colnames = ["Prediction of U2R:"])


# In[ ]:


#41 features used for cross validation
#To find Precision, recall, fscore, accuracy using cv=10

from sklearn.model_selection import cross_val_score
from sklearn import metrics

#DoS
precision=cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="precision")
print("Precision: %0.5f" % (precision.mean()))
recall=cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="recall")
print("Recall: %0.5f" % (recall.mean()))
f=cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="f1")
print("Fscore: %0.5f" % (f.mean()))
accuracy=cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#Probe
precision=cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="precision_macro")
print("Precision: %0.5f" % (precision.mean()))
recall=cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="recall_macro")
print("Recall: %0.5f" % (recall.mean()))
f=cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="f1_macro")
print("Fscore: %0.5f" % (f.mean()))
accuracy=cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#R2L
precision=cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="precision_macro")
print("Precision: %0.5f" % (precision.mean()))
recall=cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="recall_macro")
print("Recall: %0.5f" % (recall.mean()))
f=cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="f1_macro")
print("Fscore: %0.5f" % (f.mean()))
accuracy=cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#U2R
precision=cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="precision_macro")
print("Precision: %0.5f" % (precision.mean()))
recall=cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="recall_macro")
print("Recall: %0.5f" % (recall.mean()))
f=cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="f1_macro")
print("Fscore: %0.5f" % (f.mean()))
accuracy=cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#13 features that only described in rfecolname_DoS were used
X_DoS_test2=X_DoS_test[:,rfecolindex_DoS]
X_Probe_test2=X_Probe_test[:,rfecolindex_Probe]
X_R2L_test2=X_R2L_test[:,rfecolindex_R2L]
X_U2R_test2=X_U2R_test[:,rfecolindex_U2R]
X_U2R_test2.shape


# In[ ]:


#Results
#Confusion matrices that included 13 features for each of the column/category
#0:normal, 1:attack

Y_DoS_pred2 = clf_rfeDoS.predict(X_DoS_test2)
pd.crosstab(Y_DoS_test, Y_DoS_pred2, rownames = ["Actual:"], colnames = ["Prediction of DoS:"])


# In[ ]:


Y_Probe_pred2=clf_rfeProbe.predict(X_Probe_test2)
pd.crosstab(Y_Probe_test, Y_Probe_pred2, rownames = ["Actual:"], colnames=["Prediction of Probe:"])


# In[ ]:


Y_R2L_pred2=clf_rfeR2L.predict(X_R2L_test2)
pd.crosstab(Y_R2L_test, Y_R2L_pred2, rownames = ["Actual:"], colnames = ["Prediction of R2L:"])


# In[ ]:


Y_U2R_pred2=clf_rfeU2R.predict(X_U2R_test2)
pd.crosstab(Y_U2R_test, Y_U2R_pred2, rownames=["Actual:"], colnames=["Prediction of U2R:"])


# In[ ]:


#13 features used for cross validation
#To find Precision, recall, fscore, accuracy using cv=10

#DoS
precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 10, scoring = "precision")
print("Precision: %0.5f)" % (precision.mean()))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 10, scoring = "recall")
print("Recall: %0.5f" % (recall.mean()))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 10, scoring = "f1")
print("Fscore: %0.5f" % (f.mean()))
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 10, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#Probe
precision = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 10, scoring = "precision_macro")
print("Precision: %0.5f)" % (precision.mean()))
recall = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 10, scoring = "recall_macro")
print("Recall: %0.5f" % (recall.mean()))
f = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 10, scoring = "f1_macro")
print("Fscore: %0.5f" % (f.mean()))
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 10, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#R2L
precision = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 10, scoring = "precision_macro")
print("Precision: %0.5f)" % (precision.mean()))
recall = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 10, scoring = "recall_macro")
print("Recall: %0.5f" % (recall.mean()))
f = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 10, scoring = "f1_macro")
print("Fscore: %0.5f" % (f.mean()))
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 10, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#U2R
precision = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 10, scoring = "precision_macro")
print("Precision: %0.5f)" % (precision.mean()))
recall = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 10, scoring = "recall_macro")
print("Recall: %0.5f" % (recall.mean()))
f = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 10, scoring = "f1_macro")
print("Fscore: %0.5f" % (f.mean()))
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 10, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#need to check with 2,5,10,30,50
#DoS
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 2, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 5, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 10, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 30, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv = 50, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#Probe
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 2, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 5, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 10, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 30, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv = 50, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#R2L
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 2, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 5, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 10, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 30, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv = 50, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


#U2R
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 2, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 5, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 10, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 30, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:


accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv = 50, scoring = "accuracy")
print("Accuracy: %0.5f" % (accuracy.mean()))


# In[ ]:




