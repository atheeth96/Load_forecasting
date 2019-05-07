import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# for k means clustering
import pandas as pd

points_n=200
clusters_n=3
iterations_n=100
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

X=df.values
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
X=principalComponents
X = StandardScaler().fit_transform(X)
X=tf.constant(np.array(X))
centroids= tf.Variable(tf.slice(tf.random_shuffle(X), [0,0], [clusters_n,-1]))

points_expanded=tf.expand_dims(X, 0)
centroids_expanded=tf.expand_dims(centroids,1)

distances=tf.reduce_sum(tf.square(tf.subtract(points_expanded,centroids_expanded)),2)
assignments = tf.argmin(distances, 0)


means=[]
for c in range(clusters_n):
	means.append(tf.reduce_mean(tf.gather(X,tf.reshape(tf.where(tf.equal(assignments,c)),[1,-1])),reduction_indices=[1]))
new_centroids=tf.concat(means,0)

update_centroids=tf.assign(centroids,new_centroids)
init=tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for step in range(iterations_n):
		[_,centroid_values,points_value,assignment_values]=sess.run([update_centroids,centroids,X,assignments])
		print("centroids"+"\n", centroid_values)

plt.scatter(points_value[:,0],points_value[:,1],c=assignment_values,s=50,alpha=0.5)
plt.plot(centroid_values[:,0],centroid_values[:,1],'kx',markersize=15)
plt.show()

fig, ax1 = plt.subplots(figsize = (10,7))
for i in [0,1,2]:
    x=df_daily.temperatureMax.loc[df_daily['cluster']==i]
    y=df_daily.humidity.loc[df_daily['cluster']==i]
    s_wind=df_daily.windSpeed.loc[df_daily['cluster']==i]
    ax1.scatter(x, y,s = st*10,c = df_daily.cluster,label=str(i))
ax1.set_xlabel('Temperature')
ax1.set_ylabel('Humidity')
fig.legend()
plt.show()

