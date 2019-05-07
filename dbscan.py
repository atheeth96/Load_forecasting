import numpy as np
import pandas as pd
import csv
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

X=[]
df=pd.read_csv('BMSJULY2016-DEC2016.csv',dtype={"transID": int, "vltgA":float,"vltgB":float,"vltgC":float,"currA":float,	
                "currB":	float,"currC":float,	"currN":float,"F":float,	"PFA":float,
                "PFC":float,	"VTHD":float,"ITHD":float,
                "Temp":float,"finTemp":float,"sample_timestamp":object,"vltgGN":float,"eventId":float,
                    "eventTime":object,"boardTemp":float
                    })

df['eventTime'] = pd.to_datetime(df['eventTime'])
columns=['sample_timestamp','eventTime']
df.drop(columns,inplace=True,axis=1)
for columns in df.columns:
    if(len(df[columns].unique())==1):
        df.drop(columns,inplace=True,axis=1)
for columns in df.columns:
    if((df[columns].std())<=0.03):
        df.drop(columns,inplace=False,axis=1)


X=df.values
#X = StandardScaler().fit_transform(X)

X=preprocessing.normalize(X)


db = DBSCAN(eps=0.03, min_samples=5,algorithm='ball_tree').fit(X)
print(db.labels_)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print(len(labels))


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

import matplotlib.pyplot as plt

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
