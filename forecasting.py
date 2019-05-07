
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data,dropout, fully_connected
from tflearn.layers.estimator import regression
import csv
import tensorflow as tf
import os
import random
from random import shuffle
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('BMSJULY2016-DEC2016.csv',dtype={"transID": int, "vltgA":float,"vltgB":float,"vltgC":float,"currA":float,	
                "currB":	float,"currC":float,	"currN":float,"F":float,	"PFA":float,
                "PFC":float,	"VTHD":float,"ITHD":float,
                "Temp":float,"finTemp":float,"sample_timestamp":object,"vltgGN":float,"eventId":float,
                    "eventTime":object,"boardTemp":float
                    })
df['eventTime']=pd.to_datetime(df['eventTime'])
df['Weekday']=df['eventTime'].dt.weekday
df['hour']=df['eventTime'].dt.hour
df['minute']=df['eventTime'].dt.minute
df['day_of_year']=df['eventTime'].dt.dayofyear
df['Power']=df['vltgA']*df['currA']*df['PFA']+df['vltgB']*df['currB']*df['PFB']+df['vltgC']*df['currC']*df['PFC']

columns=['sample_timestamp','eventTime','vltgA','currA','PFA','vltgB','currB','PFB','vltgC','currC','PFC']#,'currN','F','VTHD','ITHD','boardTemp']
df.drop(columns,inplace=True,axis=1)

for columns in df.columns:
    if(len(df[columns].unique())==1):
        df.drop(columns,inplace=True,axis=1)

max=df['Power'].max()
min=df['Power'].min()
df = shuffle(df)
df = df[df.Weekday <= 5]

df['Quantity-1'] = df['Power'].shift(7)
df=df[['Weekday','hour','minute','day_of_year','Quantity-1','Power']]
df.dropna(subset=['Quantity-1'], inplace=True)
df.to_csv('check.csv',index=False)


train_data=[]
with open('check.csv','r') as in_file:
	
	csv_reader=csv.reader(in_file)
	next(csv_reader, None)
	for line in csv_reader:
		b=[]
		for i in range(0,5):
			x=float(line[i])
			b.append(x)
		a=float(line[5])
		train_data.append([np.array(b),a])


train = train_data[0:17082]

test=train_data[17082:-1]
train_X=np.array([i[0] for i in train])
train_X=preprocessing.normalize(train_X)
train_Y = [i[1] for i in train]
train_Y=np.reshape(train_Y,(-1,1))
train_Y=preprocessing.normalize(train_Y)


test_X = np.array([i[0] for i in test])
test_X=preprocessing.normalize(test_X)
test_Y = [i[1] for i in test]
test_Y=np.reshape(test_Y,(-1,1))
test_Y=preprocessing.normalize(test_Y)


nn=input_data(shape=[None,5],name='input')

nn=fully_connected(nn,18,activation ='relu',name='layer2')
#nn=fully_connected(nn,8,activation ='sigmoid',name='layer3')


nn=dropout(nn,0.8)

nn=fully_connected(nn,1,activation ='linear',name='final_layer')
nn=regression(nn,optimizer='adam',learning_rate=0.001,loss='mean_sabsolute',name='targets',metric='loss')
model=tflearn.DNN(nn)


model.fit({'input':train_X},{'targets':train_Y},n_epoch=10,validation_set=({'input':test_X},{'targets':test_Y}),snapshot_step=500,show_metric=True)
model.save('my_model.tflearn')
print(model.summary())

#model.load('my_model.tflearn')
op=model.predict(np.reshape([5,2,43,338,118254642],(-1,5)))
print('predicted output:',op )#op*(max-min)+min)