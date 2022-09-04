#!/usr/bin/env python
# coding: utf-8

# In[139]:



#imorting the packages
import numpy as np
import keras as ks
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
import nltk
import pandas as pd
import gensim
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Input, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Embedding
from tensorflow.keras.layers import LSTM,Embedding
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
import ktrain
from ktrain import text


# In[69]:



#Loading the dataset
headerList = ['Polarity','Unique_Number', 'Date', 'Query','Id','Tweet']
data=pd.read_csv("traindata.csv",encoding="ISO-8859-1")
data.to_csv("d1.csv", header=headerList, index=False)
  
# display modified csv file
train = pd.read_csv("d1.csv")
print('\nModified file:')
train.head()


# In[25]:



#Exploratory Data Analysis
plt.rcParams['figure.figsize']=[10,5]
plt.rcParams['figure.dpi']
sns.countplot('Polarity',data=train)
plt.title('Positive and Negative Tweets')


# In[70]:


#Random sample
df=pd.read_csv("d1.csv")
df=train.sample(10000)


# In[62]:




#data exploration and preprocessing
p_data=df[df['Polarity']==4]
n_data=df[df['Polarity']==0]


#World Cloud for positive tweets
comment_words = ''
stopwords = set(STOPWORDS)
# iterate through the csv file
for val in p_data.Tweet:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
 
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[27]:



#wordcloud for negative tweets
comment_words = ''
stopwords = set(STOPWORDS)
# iterate through the csv file
for val in n_data.Tweet:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
 
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[71]:



# replacing the value 4 by 1
df['Polarity'] = df['Polarity'].replace({4: 1})
df.info()


# In[72]:



#data preprocessing for NLP
c=[]
from nltk.stem.porter import PorterStemmer
import nltk
import re
from nltk.corpus import stopwords
ps=PorterStemmer()
for i in df.index:
    n=re.sub('[^a-zA-Z]',' ', df['Tweet'][i])
    n=n.lower()
    n=n.split()
    n=[ps.stem(word) for word in n if word not in stopwords.words('english') ]
    n=" ".join(n)
    
    c.append(n)


# In[73]:


df.Tweet=c
df.head()


# In[33]:


# Model Building
#model 1-----Random Forest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
classifier=Pipeline([("tfidf", TfidfVectorizer()) , ("classifier", RandomForestClassifier(n_estimators=100))])
X_train, X_test,y_train, y_test=train_test_split( df['Tweet'], df['Polarity'], test_size=0.30, random_state=0, shuffle=True)
y=df.iloc[:, 1].values
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[34]:


#Model 2 -----Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(df['Tweet'], df['Polarity'], test_size=0.30, random_state=0, shuffle=True)
svm=Pipeline([("tfidf",TfidfVectorizer()),("classifier",SVC(C=100,gamma='auto'))])
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[36]:


#Model 3 ----Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
y=df.iloc[:, 0].values
X_train, X_test,y_train, y_test=train_test_split(x,y, test_size=0.30, random_state=0, shuffle=True)
classi=GaussianNB()
classi.fit(X_train, y_train)
y_pred=classi.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[38]:


#Model---Logistic Regression 
x=df['Tweet']
y=df['Polarity']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
count_vect=CountVectorizer()
x_train_counts=count_vect.fit_transform(x_train)
tfidf_transformer=TfidfTransformer()
x_train_tfidf=tfidf_transformer.fit_transform(x_train_counts)
vectorizer=TfidfVectorizer()
x_train_vect=vectorizer.fit_transform(x_train)
clf=LogisticRegression(solver='lbfgs')
clf.fit(x_train_tfidf,y_train)
text_clf=Pipeline([('tfidf', TfidfVectorizer()),
                  ('clf',LogisticRegression()),])
text_clf.fit(x_train,y_train)
predictions=text_clf.predict(x_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))


# In[41]:


# Neural Networks
des=df['Tweet'].values
#Tokenize the description
des_tok=[nltk.word_tokenize(description) for description in des]
#model building
model=Word2Vec(des_tok, min_count=1)
#Predict the Word2Vec 
model.wv['love']


# In[42]:


#Finding the similar word
model.wv.most_similar('love')


# In[43]:


#Convolutional Neural Network
x_train, x_test,y_train, y_test=train_test_split(df['Tweet'],df['Polarity'], test_size=0.10)
#converting into sequence
max_vocab_size=20000
tokenizer=Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(x_train)
sequences_train=tokenizer.texts_to_sequences(x_train)
sequence_test=tokenizer.texts_to_sequences(x_test)
sequences_train
#unique tokens
word2idx=tokenizer.word_index
v=len(word2idx)
#pad sequences for equal sequences
data_train=pad_sequences(sequences_train)
t=data_train.shape[1]
#pad the test datset
data_test=pad_sequences(sequence_test,maxlen=t)
#Building the model
D=20
i=Input(shape=(t,) )
x=Embedding(v+1, D)(i)
x=Conv1D(32,3,activation='relu', padding='same')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(64,3,activation='relu', padding='same')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu', padding='same')(x)
x=GlobalMaxPooling1D()(x)
x=Dense(1,activation='sigmoid')(x)
model=Model(i,x)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accurancy'])
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
#train the model
r=model.fit(x=data_train, y=y_train,epochs=4,validation_data=(data_test, y_test))


# In[44]:


#The loss function
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='Loss')
plt.plot(r.history['val_loss'],label='Validation_ loss')
plt.legend()
plt.show()


# In[45]:


#The accuracy
import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'],label='Accuracy')
plt.plot(r.history['val_accuracy'],label='Validation_ accuracy')
plt.legend()
plt.show()


# In[74]:



# Convolutional Neural Network with L2 Regularization and Dropout
x_train, x_test,y_train, y_test=train_test_split(df['Tweet'],df['Polarity'], test_size=0.30)
#converting into sequence
max_vocab_size=20000
tokenizer=Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(x_train)
sequences_train=tokenizer.texts_to_sequences(x_train)
sequence_test=tokenizer.texts_to_sequences(x_test)
sequences_train
#unique tokens
word2idx=tokenizer.word_index
v=len(word2idx)
#pad sequences for equal sequences
data_train=pad_sequences(sequences_train)
t=data_train.shape[1]
#pad the test datset
data_test=pad_sequences(sequence_test,maxlen=t)
#Building the model with L2 Regularization and dropout
D=20
i=Input(shape=(t,) )
x=Embedding(v+1, D)(i)
x=Conv1D(32,3,kernel_regularizer=ks.regularizers.l2(0.001),activation='relu', padding='same')(x)
x=ks.layers.Dropout(0.5)(x)
x=MaxPooling1D(3)(x)
x=Conv1D(64,3,kernel_regularizer=ks.regularizers.l2(0.001),activation='relu', padding='same')(x)
x=ks.layers.Dropout(0.5)(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,kernel_regularizer=ks.regularizers.l2(0.001),activation='relu', padding='same')(x)
x=ks.layers.Dropout(0.5)(x)
x=GlobalMaxPooling1D()(x)
x=Dense(1,activation='sigmoid')(x)
model=Model(i,x)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accurancy'])
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
#train the model
r=model.fit(x=data_train, y=y_train,epochs=4,validation_data=(data_test, y_test))


# In[89]:


#Recurrent Neural Network
#Building the model
x_train, x_test,y_train, y_test=train_test_split(df['Tweet'],df['Polarity'], test_size=0.30)
D=30
M=15
i=Input(shape=(t,) )
x=Embedding(v+1, D)(i)
x=LSTM(M,return_sequences=True)(x)
x=MaxPooling1D()(x)

x=Dense(1,activation='sigmoid')(x)
model=Model(i,x)



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accurancy'])
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

#train the model
r=model.fit(x=data_train, y=y_train,epochs=4,validation_data=(data_test, y_test))


# In[90]:


#The loss function
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='Loss')
plt.plot(r.history['val_loss'],label='Validation_ loss')
plt.legend()
plt.show()


# In[91]:


#The accuracy
import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'],label='Accuracy')
plt.plot(r.history['val_accuracy'],label='Validation_ accuracy')
plt.legend()
plt.show()


# In[95]:


#Recurrent Neural Network
#Building the model with L2 Regularization and dropout
x_train, x_test,y_train, y_test=train_test_split(df['Tweet'],df['Polarity'], test_size=0.30)
D=30
M=15
i=Input(shape=(t,) )
x=Embedding(v+1, D)(i)
x=LSTM(M,kernel_regularizer=ks.regularizers.l2(0.001),return_sequences=True)(x)
x=ks.layers.Dropout(0.5)(x)
x=MaxPooling1D()(x)

x=Dense(1,activation='sigmoid')(x)
model=Model(i,x)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accurancy'])
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
#train the model
r=model.fit(x=data_train, y=y_train,epochs=3,validation_data=(data_test, y_test))


# In[96]:


#The loss function
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='Loss')
plt.plot(r.history['val_loss'],label='Validation_ loss')
plt.legend()
plt.show()


# In[97]:


#The accuracy
import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'],label='Accuracy')
plt.plot(r.history['val_accuracy'],label='Validation_ accuracy')
plt.legend()
plt.show()


# In[118]:


#LSTM
max_review_length = 80
x_train, x_test,y_train, y_test=train_test_split(df['Tweet'],df['Polarity'], test_size=0.10)
tokenizer=Tokenizer(num_words=20000, lower=True, oov_token='<UNK>')
tokenizer.fit_on_texts(x_train)
word_index=tokenizer.word_index
tokenizer=Tokenizer(num_words=20000, lower=True, oov_token='<UNK>')
tokenizer.fit_on_texts(x_test)
word_index=tokenizer.word_index
d_seq_train=tokenizer.texts_to_sequences(x_train)
d_seq_test=tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(d_seq_train, truncating='pre', padding='pre', maxlen=max_review_length)
x_test = sequence.pad_sequences(d_seq_test, truncating='pre', padding='pre', maxlen=max_review_length)

model=Sequential()
model.add(Embedding(input_dim=50000, output_dim=10, mask_zero=True))
model.add(LSTM(units=50))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size=32
r=model.fit(x_train, y_train, batch_size=batch_size,epochs=4, validation_data=(x_test,y_test))


# In[119]:


#The loss function
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='Loss')
plt.plot(r.history['val_loss'],label='Validation_ loss')
plt.legend()
plt.show()


# In[120]:


#The accuracy
import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'],label='Accuracy')
plt.plot(r.history['val_accuracy'],label='Validation_ accuracy')
plt.legend()
plt.show()


# In[128]:





# In[136]:



#Bi directional LSTM
x_train, x_test,y_train, y_test=train_test_split(df['Tweet'],df['Polarity'], test_size=0.50)
max_review_length = 80
x_train, x_test,y_train, y_test=train_test_split(df['Tweet'],df['Polarity'], test_size=0.10)
tokenizer=Tokenizer(num_words=20000, lower=True, oov_token='<UNK>')
tokenizer.fit_on_texts(x_train)
word_index=tokenizer.word_index
tokenizer=Tokenizer(num_words=20000, lower=True, oov_token='<UNK>')
tokenizer.fit_on_texts(x_test)
word_index=tokenizer.word_index
d_seq_train=tokenizer.texts_to_sequences(x_train)
d_seq_test=tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(d_seq_train, truncating='pre', padding='pre', maxlen=max_review_length)
x_test = sequence.pad_sequences(d_seq_test, truncating='pre', padding='pre', maxlen=max_review_length)

model=Sequential()
model.add(Embedding(input_dim=50000, output_dim=10, mask_zero=True))
model.add(Bidirectional(LSTM(units=50)))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size=32
r=model.fit(x_train, y_train, batch_size=batch_size,epochs=4, validation_data=(x_test,y_test))


# In[ ]:





# In[137]:


#The loss function
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='Loss')
plt.plot(r.history['val_loss'],label='Validation_ loss')
plt.legend()
plt.show()


# In[138]:


#The accuracy
import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'],label='Accuracy')
plt.plot(r.history['val_accuracy'],label='Validation_ accuracy')
plt.legend()
plt.show()


# In[ ]:


#Sentiment classification using BERT
get_ipython().system('pip install ktrain')


# In[143]:


#Random sample
import pandas as pd
df=pd.read_csv("d1.csv")
df=train.sample(100)
# replacing the value 4 by 1
df['Polarity'] = df['Polarity'].replace({4: 1})
df.info()


# In[ ]:


#data preprocessing for NLP
c=[]
from nltk.stem.porter import PorterStemmer
import nltk
import re
from nltk.corpus import stopwords
ps=PorterStemmer()
for i in df.index:
    n=re.sub('[^a-zA-Z]',' ', df['Tweet'][i])
    n=n.lower()
    n=n.split()
    n=[ps.stem(word) for word in n if word not in stopwords.words('english') ]
    n=" ".join(n)
    
    c.append(n)


# In[ ]:


df.Tweet=c
df.head()


# In[ ]:


#This might take some time


# In[ ]:


#values of original dataframe
d1_train = df.sample(frac = 0.5)
 
# Creating dataframe with
# rest of the 50% values
d2_test = df.drop(d1_train.index)
(X_train, y_train),(X_test,y_test),preproc=text.texts_from_df(train_df=d1_train,
                                                           text_column='Tweet',
                                                           label_columns='Polarity',
                                                           val_df=d2_test,
                                                           maxlen=50,
                                                           preprocess_mode='bert')
model=text.text_classifier(name='bert',
                        train_data=(X_train, y_train),
                        preproc=preproc)
learner=ktrain.get_learner(model=model,train_data=(X_train,y_train),
                val_data=(X_test,y_test),
                batch_size=64)
learner.fit_onecycle(lr=2e-5,epochs=1)


# In[ ]:





# In[146]:





# In[148]:





# In[ ]:





# In[167]:





# In[ ]:





# In[169]:





# In[170]:





# In[171]:





# In[172]:





# In[ ]:





# In[ ]:





# In[160]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




