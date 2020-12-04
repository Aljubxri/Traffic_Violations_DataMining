#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('Traffic-Violations6K.csv')


# In[2]:


#Dropped all non operated vehicles 
df = df[df.VehicleType !="28 - Other"]
df = df[df.VehicleType !="29 - Unknown"]

#Dropped all missing rows
list1 = {"Geolocation", "Driver City", "Driver State", "DL State", "Arrest Type", "Geolocation", 'Color', "Article"}
for x in list1:
    df = df[df[x].notnull()]
#Dropping notneeded Attributes
df = df.drop(columns = ['Article', 'Date Of Stop', 'SubAgency', 'Location', 'Latitude', 'Longitude', 'Driver City', 'Driver State', 'DL State','Geolocation', 'Agency', 'State'])
 


# In[30]:


#Cleaning:


# In[3]:


#cleans the time column so only shows hours
df['Time Of Stop'] = df['Time Of Stop'].str[:2]
df['Time Of Stop'] = df['Time Of Stop'].str.extract('(\d+)', expand=False)


#bins the hours
pd.cut(df['Time Of Stop'].astype(int),bins=6 )
cut_labels = ['Night','Morning','Afternoon','Evening']
cut_bins = [-1,5,12,17,24]
df['Time Of Stop'] = pd.cut(df['Time Of Stop'].astype(int), bins=cut_bins, labels=cut_labels)


# In[4]:


#cleans the Makes of the vehicles 
df= df[df.Make !="2004"]
df.loc[df['Make'] == 'ACUR', "Make"] = "ACURA"
df.loc[df['Make'] == 'ACCUR', "Make"] = "ACURA"

df.loc[df['Make'] == 'BUIC', "Make"] = "BUICK"

df.loc[df['Make'] == 'CADI', "Make"] = "CADILLAC"
df.loc[df['Make'] == 'CADILAC', "Make"] = "CADILLAC"

df.loc[df['Make'] == 'CHEV', "Make"] = "CHEVROLET"
df.loc[df['Make'] == 'CHEVORLET', "Make"] = "CHEVROLET"
df.loc[df['Make'] == 'CHRY', "Make"] = "CHEVROLET"
df.loc[df['Make'] == 'CHEVY', "Make"] = "CHEVROLET"

chrys = 'CHRYLSER', 'CHRYSLAR', 'CHRYSLER','CHRYSTLER', 'CHYSLER', 'CRUZ','CRYSTLER'
for x in chrys:
    df.loc[df['Make'] == x, "Make"] = "CHRYSLER"

df.loc[df['Make'] == 'DODG', "Make"] = "DODGE"

df.loc[df['Make'] == 'HINDA', "Make"] = "HONDA"
df.loc[df['Make'] == 'HOND', "Make"] = "HONDA"

hyundai = 'HUYNDAI', 'HYANDAI','HYNDAI', 'HYUN', 'HYUNDIA'
for x in hyundai:
    df.loc[df['Make'] == x, "Make"] = "HYUNDAI"

inf = 'INF','INFI','INFINITY','INIF','INIFINITI'
for x in inf:
    df.loc[df['Make'] == x, "Make"] = "INFINITI"

ISU = 'ISU','ISUZUE', 'IZUZU'
for x in ISU:
    df.loc[df['Make'] == x, "Make"] = "ISUZU"

lex = 'LEX', 'LEXS', 'LEXU'
for x in lex:
    df.loc[df['Make'] == x, "Make"] = "LEXUS"

mazda = 'MAZA', 'MAZADA', 'MAZD'
for x in mazda:
    df.loc[df['Make'] == x, "Make"] = "MAZDA"
    
MB = 'MB', 'MERC','MERCEDES BENZ', 'MERCEDEZ', 'M BENZ', "MERZ"
for x in MB:
    df.loc[df['Make'] == x, "Make"] = "MERCEDES"

mitsu = 'MITIS', 'MITISHBSHI' ,'MITS','MITSUBUSHI', 'MITTS', 'MITZ'
for x in mitsu:
    df.loc[df['Make'] == x, "Make"] = "MITSUBISHI"

nis = 'NIS','NISS','NISSAB','NISSIAN'
for x in nis:
    df.loc[df['Make'] == x, "Make"] = "NISSAN"

df.loc[df['Make'] == "PORS", "Make"] = "PORSCHE"

satu = 'SAT', 'SATU', 'SATURRN', 'STRN'
for x in satu:
    df.loc[df['Make'] == x, "Make"] = "SATURN"

subie = 'SUB', 'SUBA', 'SUBRARU', 'SUBURU'
for x in subie:
    df.loc[df['Make'] == x, "Make"] = "SUBARU"

toyo = 'TOY', 'TOYATA' ,'TOYO', 'TOYOT', 'TOYOTOA', 'TOYOVAL2014', 'TOYOYA','TOYPTA' ,'TOYT', 'TOYTA'
for x in toyo:
    df.loc[df['Make'] == x, "Make"] = "TOYOTA"

vw = 'VOLK', 'VOLKD','VOLKS' ,'VOLKSWAGON','VW', "WV"
for x in vw:
    df.loc[df['Make'] == x, "Make"] = "VOLKSWAGEN"

vol = 'VOLV','VOLV0','VOLVA' 
for x in vol:
    df.loc[df['Make'] == x, "Make"] = "VOLVO"

LD = 'LAD ROVER', 'LANDROVER', 'LNDR'
for x in LD:
    df.loc[df['Make'] == x, "Make"] = "LAND ROVER"


remove = 'PTRB','PONT','2004','BIGT', 'GEO', 'GMC' ,'GRUMAN','KAW' , 'KEN', 'LIINCOLN','LINC' ,'OLDS' , 'PETE' , 'PONT' 'PTRB','SCIO','SILVER', 'SUZI', 'SUZU' 
for x in remove:
    df= df[df.Make !=x]


# In[5]:


#Cleans Arrest type into 4 groups, could be used as target variables
df.loc[df['Arrest Type'] == 'A - Marked Patrol' , "Arrest Type"] = 'Marked Patrol'
df.loc[df['Arrest Type'] == 'Q - Marked Laser' , "Arrest Type"] = 'Marked Radar'
df.loc[df['Arrest Type'] == 'B - Unmarked Patrol' , "Arrest Type"] = 'Unmarked Patrol'
df.loc[df['Arrest Type'] == 'S - License Plate Recognition' , "Arrest Type"] = 'Marked Patrol'
df.loc[df['Arrest Type'] == 'M - Marked (Off-Duty)' , "Arrest Type"] = 'Marked Patrol'
df.loc[df['Arrest Type'] == 'E - Marked Stationary Radar' , "Arrest Type"] = 'Marked Radar'
df.loc[df['Arrest Type'] == 'L - Motorcycle' , "Arrest Type"] = 'Marked Patrol'
df.loc[df['Arrest Type'] == 'O - Foot Patrol' , "Arrest Type"] = 'Marked Patrol'
df.loc[df['Arrest Type'] == 'D - Unmarked VASCAR' , "Arrest Type"] = 'Unmarked Patrol'
df.loc[df['Arrest Type'] == 'F - Unmarked Stationary Radar' , "Arrest Type"] = 'Unmarked Radar'
df.loc[df['Arrest Type'] == 'G - Marked Moving Radar (Stationary)' , "Arrest Type"] = 'Marked Radar'
df.loc[df['Arrest Type'] == 'I - Marked Moving Radar (Moving)' , "Arrest Type"] = 'Marked Radar'
df.loc[df['Arrest Type'] == 'H - Unmarked Moving Radar (Stationary)' , "Arrest Type"] = 'Unmarked Radar'
df.loc[df['Arrest Type'] == 'C - Marked VASCAR' , "Arrest Type"] = 'Marked Patrol'



# In[ ]:


#Visulaziations


# In[6]:


make_counts = df.groupby(['Make']).size()
make_names = df['Make'].unique()
make_names.sort()
fig = go.Figure([go.Bar(x=make_names, y=make_counts)])
fig.show()


# In[7]:


time_counts = df.groupby(['Time Of Stop']).size()
time_names = ['Night', 'Morning', 'Afternoon','Evening']
fig = go.Figure([go.Bar(x=time_names, y=time_counts,marker_color=['purple','lightblue','lightsalmon', 'lightyellow'])])
fig.show()


# In[8]:


value = df['Belts'].value_counts()
Belt = df['Belts'].unique()
colors = ['darkorange', 'lightblue']
fig = go.Figure(data=[go.Pie(labels=Belt, values=value, hole=.5)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+label+value', textfont_size=15,
                  marker=dict(colors=colors, line=dict(color='black', width=2)))
fig.update_layout(
    title_text="% of Seatbelts used in violations",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='2012 - 2016', x=0.50, y=0.5, font_size=20, showarrow=False)])


fig.show()


# In[9]:


describ = df['Description'].value_counts()
sliced_desc = describ[:10]
index_ = ['Traffic light violation', 'Suspended Registration', 'Suspended License', 'Failure to display registeration', 'Failure to display license to officer',
          'Expired registration plate', 'Impaired by alcohol', 'DUI', 'Suspended license', 'No seatbelt'] 
sliced_desc.index = index_
sliced_desc

fig = go.Figure(data=[go.Pie(labels=index_, values=sliced_desc, hole=.5)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', textfont_size=18,
                  marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title_text="Top 10 most common Traffic Violations 2012-2016")
fig.update(layout_showlegend=False)

fig.show()


# In[11]:


dfz = df
list11 = list(df)

dfz = dfz.drop(columns=list11, axis = 1)
dfz['Belts'] = df['Belts'].eq('Yes').mul(1)
dfz['Contributed To Accident'] = df['Contributed To Accident'].eq('Yes').mul(1)
dfz['Personal Injury'] = df['Personal Injury'].eq('Yes').mul(1)
dfz['Property Damage'] = df['Property Damage'].eq('Yes').mul(1)
dfz['Fatal'] = df['Fatal'].eq('Yes').mul(1)
dfz['Male'] = df['Gender'].eq('M').mul(1)
dfz['Female'] = df['Gender'].eq('F').mul(1)



sns.heatmap(dfz.corr(), annot = True,cmap="Greens") 



# In[ ]:


#One hot encoding


# In[12]:


arrest1 = pd.get_dummies(df['Arrest Type'])
stop1 = pd.get_dummies(df['Time Of Stop'])
violation1 = pd.get_dummies(df['Violation Type'])




hotdf = pd.concat([dfz, arrest1, stop1, violation1], axis=1)
hotdf["Gender"] = df["Gender"]

#1 is marked patrol, 2 is unmarked patrol, 3 is marked radar, 4 is unmarked radar
#one_hot.loc[one_hot['Arrest'] == 'Marked Patrol' , "Arrest"] = 1
#one_hot.loc[one_hot['Arrest'] == 'Unmarked Patrol' , "Arrest"] = 2
#one_hot.loc[one_hot['Arrest'] == 'Marked Radar' , "Arrest"] = 3
#one_hot.loc[one_hot['Arrest'] == 'Unmarked Radar' , "Arrest"] = 4


hotdf.loc[hotdf['Gender'] == 'M', "Gender"] = "1"
hotdf.loc[hotdf['Gender'] == 'F', "Gender"] = "0"
hotdf.head()


# Linear Regression

# In[13]:


#features given
X = df[['Belts', 'Violation Type', 'Arrest Type', 'Time Of Stop']]
#what I want to predict
Y = hotdf['Gender']
X = pd.get_dummies(data=X, drop_first=True)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 40)
regr = linear_model.LinearRegression() # Do not use fit_intercept = False if you have removed 1 column after dummy encoding
regr.fit(X_train, Y_train)
predicted = regr.predict(X_test)

test_set = r2_score(Y_test, predicted)
print (test_set)


# In[26]:


#Elbow Method


# In[28]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.show()


# In[ ]:


#Kmeans


# In[24]:


from sklearn.cluster import KMeans
km = KMeans(
    n_clusters=3, init='random',
    n_init=20, max_iter=300, 
    tol=1e-04, random_state=0
)
X = hotdf
y_km = km.fit_predict(X)


# In[25]:


# plot the 3 clusters
plt.scatter(
    X.iloc[y_km == 0, 0], X.iloc[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X.iloc[y_km == 1, 0], X.iloc[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X.iloc[y_km == 2, 0], X.iloc[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


# In[ ]:


#Logistical Regression


# In[29]:


from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Y = hotdf['Night']
#X = hotdf
X = df[['Belts', 'Violation Type', 'Arrest Type', 'Time Of Stop']]
#what I want to predict
Y = hotdf['Gender']
X = pd.get_dummies(data=X, drop_first=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 40)

# Standardizing the features
X_train_scale=scale(X_train)
X_test_scale=scale(X_test)
# Fitting the logistic regression model
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)
# Checking the models accuracy
accuracy_score(Y_test,log.predict(X_test_scale))

