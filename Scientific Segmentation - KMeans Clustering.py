#!/usr/bin/env python
# coding: utf-8

# # <center> Scintific Segmentation (K-Means Clustering) <center>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.stats as stats
import pandas_profiling   #need to install using anaconda prompt (pip install pandas_profiling)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA  


# In[8]:


# reading data into dataframe
telco= pd.read_csv("C:/telco_csv.csv")


# In[9]:


telco.head()


# In[10]:


telco.info()


# In[11]:


telco_new = pd.get_dummies(telco, columns=['region'], drop_first=True, prefix='region') #one hot encoding


# In[12]:


telco_new = pd.get_dummies(telco_new, columns=['custcat'], drop_first=True, prefix='cust_cat') #one hot encoding


# In[13]:


#Handling missings - Method2
def Missing_imputation(x):
    x = x.fillna(x.median())
    return x

telco_new=telco_new.apply(lambda x: Missing_imputation(x))


# In[14]:


#Handling Outliers - Method2
def outlier_capping(x):
    #x = x.clip_upper(x.quantile(0.99))
    #x = x.clip_lower(x.quantile(0.01))
    x = x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
    return x

telco_new=telco_new.apply(lambda x: outlier_capping(x))


# In[15]:


telco_new.columns


# In[10]:


report = pandas_profiling.ProfileReport(telco_new)


# In[11]:


report.to_file('report.html')


# In[16]:


telco_new.drop(columns = ['wireless', 'equip'], axis=1, inplace=True)


# In[17]:


telco_new.columns


# In[18]:


telco_new.apply(lambda x: x.std()/x.mean())


# ## Dimension Reduction - Principle Component Analysis (PCA)

# In[19]:


#standardize the data

sc = StandardScaler()
sc = sc.fit(telco_new)
telco_new_std = pd.DataFrame(sc.transform(telco_new), columns = telco_new.columns)


# In[20]:


telco_new_std


# In[21]:


#unsupervised: variable reduction techniques
# VIF, Correlation matrics, PCA

pca_model = PCA(n_components = 31)
pca_model = pca_model.fit(telco_new_std)


# In[22]:


sum(pca_model.explained_variance_ )


# In[23]:


pca_model.explained_variance_     #Eigen values


# In[24]:


pca_model.explained_variance_ratio_


# In[25]:


np.cumsum(pca_model.explained_variance_ratio_)


# In[26]:


#Criteria to choose number of compoents
  #  1. Cumulative should be more than 75%
  #  2. Individual component should explain more than 0.8 variancce
#number of components = 10  


# In[27]:


pca_model = PCA(n_components = 10)
pca_model = pca_model.fit(telco_new_std)


# In[28]:


PCs = pd.DataFrame(pca_model.transform(telco_new_std), columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])


# In[29]:


PCs     # you are able to reduce dimesnisons of 65% (from 31 to 10 ) by loosing the information about 24% - Dimension Reduction


# In[30]:


#Build segmentation - Input: PCs


# In[31]:


#Loading matrics

Loadings =  pd.DataFrame((pca_model.components_.T * np.sqrt(pca_model.explained_variance_)).T,columns=telco_new.columns).T

Loadings.columns= ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10' ]


# In[32]:


Loadings.to_csv('Loadings.csv')


# In[33]:


Final_list = ['tollmon',
'voice',
'longmon',
'employ',
'multline',
'reside',
'region_2',
'income',
'retire',
'gender'
]


# In[34]:


#seg_input = telco_new_std[Final_list]  # variable reduction output
seg_input = PCs   #Dimension Reduction output


# In[35]:


#KMeans Clustering   input: standardized data, number of cluster

km_3 = KMeans(n_clusters=3, random_state=123).fit(seg_input)


# In[36]:


km_3.labels_   #Segment labels


# In[37]:


km_3.cluster_centers_


# In[38]:


km_3 = KMeans(n_clusters=3, random_state=123).fit(seg_input)
km_4 = KMeans(n_clusters=4, random_state=123).fit(seg_input)
km_5 = KMeans(n_clusters=5, random_state=123).fit(seg_input)
km_6 = KMeans(n_clusters=6, random_state=123).fit(seg_input)
km_7 = KMeans(n_clusters=7, random_state=123).fit(seg_input)
km_8 = KMeans(n_clusters=8, random_state=123).fit(seg_input)


# In[39]:


telco_new['Cluster_3'] = km_3.labels_
telco_new['Cluster_4'] = km_4.labels_
telco_new['Cluster_5'] = km_5.labels_
telco_new['Cluster_6'] = km_6.labels_
telco_new['Cluster_7'] = km_7.labels_
telco_new['Cluster_8'] = km_8.labels_


# In[40]:


telco_new.head()


# In[41]:


#To finding the optimal solution (optimal value of k), we follow below approaches
    #1. Using SC metrics/Elbow Analysis
    #2. Using profiling
    #3. Best practices

#Choosing best solution (optimal solution) - Identifying best value of K
#1. Metrics	
    a. Silhoutte coeficient	between -1 & 1	
        #		Closer to 1, segmentation is good	
        #		Closer to-1, segmentation is bad	
	b. Pseudo F-value	

#2. Profiling of segments			
#3. Best practices			
#	Segment distribution	4%-40%	
#	Strategy can be implementable or not	
# ### Silhoutte Score

# In[42]:


silhouette_score(seg_input, km_4.labels_)


# In[43]:


# calculate SC for K=3 through K=12
k_range = range(3, 8)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=123)
    km.fit(seg_input)
    scores.append(silhouette_score(seg_input, km.labels_))


# In[44]:


# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)


# In[ ]:


scores


# In[ ]:


#based on sc score, the solution K=3, the second solution can be k=4


# ### Elbow Analysis 

# In[44]:


cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( seg_input )
    cluster_errors.append( clusters.inertia_ )


# In[45]:


clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:10]


# In[46]:


# allow plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# ### Note:
# - The elbow diagram shows that the gain in explained variance reduces significantly from 3 to 4 to 5. So, optimal number of clusters could either 4 or 5. 
# - The actual number of clusters chosen can be finally based on business context and convenience of dealing with number of segments or clusters.

# ### Segment Distribution

# In[47]:


telco_new.tenure.count()


# In[48]:


telco_new.Cluster_3.value_counts().sort_index()/sum(telco_new.Cluster_3.value_counts())


# In[49]:


#K=3, K=4
seg_dist = pd.concat([pd.Series(telco_new.Cluster_3.count())/telco_new.Cluster_3.count(),
           telco_new.Cluster_3.value_counts().sort_index()/sum(telco_new.Cluster_3.value_counts()),
           telco_new.Cluster_4.value_counts().sort_index()/sum(telco_new.Cluster_4.value_counts()),
           telco_new.Cluster_5.value_counts().sort_index()/sum(telco_new.Cluster_5.value_counts()),
           telco_new.Cluster_6.value_counts().sort_index()/sum(telco_new.Cluster_6.value_counts()),
           telco_new.Cluster_7.value_counts().sort_index()/sum(telco_new.Cluster_7.value_counts()),
           telco_new.Cluster_8.value_counts().sort_index()/sum(telco_new.Cluster_8.value_counts())])


# In[50]:


agg = pd.concat([telco_new.apply(np.mean).T,
           telco_new.groupby('Cluster_3').apply(np.mean).T,
           telco_new.groupby('Cluster_4').apply(np.mean).T,
           telco_new.groupby('Cluster_5').apply(np.mean).T,
           telco_new.groupby('Cluster_6').apply(np.mean).T,
           telco_new.groupby('Cluster_7').apply(np.mean).T,
           telco_new.groupby('Cluster_8').apply(np.mean).T], axis=1)
           


# In[51]:


pd.DataFrame(seg_dist).T


# In[52]:


profiling = pd.concat([pd.DataFrame(seg_dist).T, agg],axis=0)


# In[53]:


profiling.columns = ['overall',
                    'KM3_1', 'KM3_2', 'KM3_3',
                    'KM4_1', 'KM4_2', 'KM4_3','KM4_4',
                    'KM5_1', 'KM5_2', 'KM5_3','KM5_4', 'KM5_5',
                    'KM6_1', 'KM6_2', 'KM6_3','KM6_4', 'KM6_5', 'KM6_6',
                    'KM7_1', 'KM7_2', 'KM7_3','KM7_4', 'KM7_5', 'KM7_6','KM7_7',
                    'KM8_1', 'KM8_2', 'KM8_3','KM8_4', 'KM8_5', 'KM8_6', 'KM8_7', 'KM8_8']


# In[54]:


profiling.to_csv('profiling.csv')


# ### Predicting segment for new data

# In[2]:


new = pd.read_csv('C:/Telco_new_cust.csv')


# In[3]:


new


# In[4]:


new.drop(columns = ['wireless', 'equip'], axis=1, inplace=True)


# In[5]:


new1 = pd.get_dummies(new, columns=['region'], drop_first=True, prefix='region') #one hot encoding

new1 = pd.get_dummies(new1, columns=['custcat'], drop_first=True, prefix='cust_cat') #one hot encoding


# In[6]:


new1


# In[45]:


new_std = pd.DataFrame(sc.transform(new1), columns = new1.columns)


# In[46]:


new_std


# In[47]:


new_PCs = pd.DataFrame(pca_model.transform(new_std), columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])


# In[49]:


new_PCs


# In[54]:


new['pred_seg'] = km_4.predict(new_PCs)


# In[55]:


new


# In[52]:


km_4.labels_


# In[63]:


new


# In[64]:


new.pred_seg.value_counts()


# In[ ]:




