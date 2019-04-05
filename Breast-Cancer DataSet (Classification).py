#!/usr/bin/env python
# coding: utf-8

# In[1]:



# pyhton module eklendi
import sklearn 
  
# odevde verilen dataset yuklendi
from sklearn.datasets import load_breast_cancer 


# In[2]:


# Datasetimi yukledim
data = load_breast_cancer() 


# In[3]:


# datami duzenledim labeller ekledim 
label_names = data['target_adi'] 
labels = data['target'] 
feature_names = data['feature_adi'] 
features = data['data'] 


# In[4]:


# dataya bakmak
print(label_names) 


# In[5]:


print(labels)


# In[6]:


print(feature_names)  #sadece merak feauture yani columnlara baktim


# In[7]:


# functionu importladim
from sklearn.model_selection import train_test_split 
  
# Datayi train ve test set olarak ayirdim test 33%- 67%
train, test, train_labels, test_labels = train_test_split(features, labels, 
                                       test_size = 0.33, random_state = 42) 


# In[8]:


# ML modelimi importladim
from sklearn.naive_bayes import GaussianNB 
  
#classifierin kurulmasi
gnb = GaussianNB() 
  
# training modelimi sinifladim
model = gnb.fit(train, train_labels) 


# In[9]:


# tahmin yurutme
predictions = gnb.predict(test) 
  
# tahmini goruntuleme
print(predictions)


# In[10]:


# dogruluk olcum functionunu import ettim
from sklearn.metrics import accuracy_score 
  
# test setimle predictionum ne kadar ortustu dogruluk orani 94.15% cikti basarili bir odev
print(accuracy_score(test_labels, predictions)) 


# In[ ]:




