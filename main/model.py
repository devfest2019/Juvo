#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ## Load Datasets

# In[2]:


# load dataset
df = pd.read_csv('dataset/credit_train.csv',header=0)


# In[3]:


df = df.iloc[:100000]


# In[4]:


df.head(n=10)


# In[5]:


df.tail(n=10)


# ## Drop Columns

# In[6]:


# Drop irrelevant columns
drop_cols = ['Loan ID', 'Customer ID', 'Purpose','Monthly Debt','Months since last delinquent', 'Credit Score',
             'Number of Open Accounts','Maximum Open Credit','Bankruptcies','Tax Liens']


# In[7]:


df = df.drop(drop_cols, axis=1)
df.head()


# ## Drop NaN

# In[8]:


len(df)


# In[9]:


df.isnull().sum()


# In[10]:


df = df[df['Annual Income'].notnull()]
len(df)


# In[11]:


df.isnull().sum()


# ## Remove Outliers

# In[12]:


# Locate outliers in Current Loan Amount
print(len(df.loc[df['Current Loan Amount'] == 99999999]))


# In[13]:


# remove outliers
df = df.loc[df['Current Loan Amount'] != 99999999]
print(len(df))


# ## Label Encoding

# In[14]:


df.head()


# In[15]:


# Label-encode data
df.loc[df['Loan Status'] == 'Fully Paid', 'Loan Status'] = 1
df.loc[df['Loan Status'] == 'Charged Off', 'Loan Status'] = 0
df.loc[df['Term'] == 'Short Term', 'Term'] = 0
df.loc[df['Term'] == 'Long Term', 'Term'] = 1
df.loc[df['Home Ownership'] != 'Own Home', 'Home Ownership'] = 0
df.loc[df['Home Ownership'] == 'Own Home', 'Home Ownership'] = 1


# In[16]:


df.head()


# ## One-Hot Encoding

# In[17]:


# one-hot encoding
def onehot_encode(df, feature):
    encoded_df = pd.get_dummies(df[feature], prefix = feature, dummy_na=True)
    
    # concatenate original training data and encoded data, drop sex feature
    df = pd.concat([df,encoded_df],axis=1)
    df = df.drop([feature], axis=1)
    return df

# One-hot Encode all the columns
def onehot_encode_cols(df, cols):
    for col in cols:
        df = onehot_encode(df, col)
    return df

# visualize column headers
df.columns.values


# In[18]:


onehot_cols = ['Years in current job']
df = onehot_encode_cols(df, onehot_cols)


# In[19]:


print(len(df.columns.values))


# ## Normalize Data

# In[20]:


normalize_cols = ['Current Loan Amount','Annual Income',
                  'Years of Credit History','Number of Credit Problems','Current Credit Balance']


# In[21]:


# save means and stds
mean_dict = {}
std_dict = {}
for col in normalize_cols:
    mean_dict[col] = df[col].mean()
    std_dict[col] = df[col].std()


# In[22]:


df.head()


# In[23]:


df.head()


# In[24]:


def normalize_data(df, features):
    normalizer = StandardScaler()
    df[features]=normalizer.fit_transform(df[features])
    return df
df = normalize_data(df, normalize_cols)


# In[25]:


df.head()


# ## Split Data into Features and Target

# In[26]:


Y, X = df['Loan Status'], df.drop(['Loan Status'], axis=1)


# In[27]:


Y.head()


# In[28]:


X.head()


# ## Split Data into Train, Valid, Test

# In[29]:


len(X), len(Y)


# In[30]:


# specify the ratio of validation data set
test_ratio = 0.20
X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(X, Y, test_size=test_ratio)
len(X_train_valid), len(X_test)


# In[31]:


# specify the ratio of validation data set
valid_ratio = 0.20
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, test_size=valid_ratio)
len(X_train), len(X_valid), len(X_test)


# ## Neural Network Functions

# In[32]:


def plot_history(history, model=None, train_accuracy=None, test_accuracy=None, neurons=None, dropout_percentage=None, epoch=None):
    '''
    from 1jinwoo/ClassiPy project
    Plots history from training result
    '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    if model is not None:
        plt.title(model+', train accuracy = '+str(train_accuracy)+', test_accuracy = '+str(test_accuracy))
    else:
        plt.title('Training and validation loss')
    plt.legend()
    if model is not None:
        plt.savefig('images/'+model+'Neurons'+str(neurons)+' Dropout'+str(dropout_percentage)+' Epoch'+str(epoch)+'.jpg')
    else:
        plt.show()
    plt.close()
    
# early stopping is used to prevent overtraining -> we will stop the training "early" if it has reached maximum accuracy
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',restore_best_weights=True)
callbacks_list = [early_stopping]

print('functions loaded')


# ## Neural Network Training

# In[33]:


X_train.shape[1]


# In[34]:


# specify the input and output dimensions of the neural network
input_dim = X_train.shape[1]
output_dim = 1


# In[35]:


# def simpleNN():
#     model = Sequential()
#     model.add(Dense(input_dim, input_dim=input_dim, activation='relu', use_bias=True))
#     model.add(Dense(10, activation='relu', use_bias=True))
#     model.add(Dropout(rate=0.3))
#     model.add(Dense(output_dim, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['accuracy'])
#     return model
# 
# 
# # In[36]:
# 
# 
# model = simpleNN()
# history = model.fit(X_train, Y_train,
#                     epochs=10,
#                     verbose=1,
#                     validation_data = (X_valid, Y_valid),
#                     callbacks=callbacks_list,
#                     batch_size=8)
# print(model.summary())
# 
# loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
# print('Training Accuracy: {:.4f}'.format(accuracy))
# loss, accuracy = model.evaluate(X_valid, Y_valid, verbose=False)
# print('Validation Accuracy: {:.4f}'.format(accuracy))
# loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
# print('Testing Accuracy: {:.4f}'.format(accuracy))
# 
# plot_history(history)


# In[37]:


'''
seed = 7
np.random.seed(seed)
estimator = KerasClassifier(build_fn=simpleNN,
                            epochs=10,
                            batch_size=32,
                            verbose=1)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print('K-fold Validation Results: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))
'''


# ## API
def predict(d):
    x = pd.DataFrame(np.array([[0]*19]),columns = list(X.columns.values))
    print("Input dict: " + str(d))
    
    income = d['income']
    amount = d['amount']
    term = d['term']
    credit_balance = d['credit_balance']
    years_in_job = d['years_in_job']
    years_credit_history = d['years_credit_history']
    number_credit_problems = d['number_credit_problems']
    home = d['home'] if 'home' in d.keys() else False
    
    print("Mean_dict: " + str(mean_dict))
    print("Mean_dict type: " + str(type(mean_dict)))
    
    x['Current Loan Amount'] = (amount - mean_dict['Current Loan Amount'])/std_dict['Current Loan Amount']
    x['Term'] = 1 if term >= 12 else 0
    x['Annual Income'] = (income - mean_dict['Annual Income'])/std_dict['Annual Income']
    x['Home Ownership'] = 1 if home else 0
    x['Years of Credit History'] = (years_credit_history - mean_dict['Years of Credit History'])/std_dict['Years of Credit History']
    x['Number of Credit Problems'] = (number_credit_problems - mean_dict['Number of Credit Problems'])/std_dict['Number of Credit Problems']
    x['Current Credit Balance'] = (credit_balance - mean_dict['Current Credit Balance'])/std_dict['Current Credit Balance']
    if years_in_job < 1:
        x['Years in current job_< 1 year'] = 1
    if years_in_job == 1:
        x['Years in current job_1 year'] = 1
    if years_in_job == 2:
        x['Years in current job_2 years'] = 1
    if years_in_job == 3:
        x['Years in current job_3 years'] = 1
    if years_in_job == 4:
        x['Years in current job_4 years'] = 1
    if years_in_job == 5:
        x['Years in current job_5 years'] = 1
    if years_in_job == 6:
        x['Years in current job_6 years'] = 1
    if years_in_job == 7:
        x['Years in current job_7 years'] = 1
    if years_in_job == 8:
        x['Years in current job_8 years'] = 1
    if years_in_job == 9:
        x['Years in current job_9 years'] = 1
    if years_in_job >= 10:
        x['Years in current job_10+ years'] = 1
    return x    


# In[40]:


d = {'income': 500,
    'amount': 50,
    'term' : 24,
    'credit_balance' : 0,
    'years_in_job' : 3,
    'years_credit_history' : 40,
    'number_credit_problems' : 0,
    'home' : True}
predict(d)


# In[42]:


X


# In[ ]:




