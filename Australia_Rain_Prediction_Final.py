#!/usr/bin/env python
# coding: utf-8

# # Rain in Australia
# 
# 
# ### Table of Contents
# 
# #### 1. **Information**
#     - Reason for Choosing this Dataset ?**
#     - Details
#     - Objective
# 
# #### 2. **Loading Dataset**
#     - Importing packages
#     - Reading Data
#     - Shape of data
#     - Dtype
# 
# #### 3. **Data Cleansing & EDA**
#     - Checking Null values
#     - Descriptive Statistics
#     - Viz. (Phase 1)
#     - Month & Year Extraction
#     - Viz. (Phase 2)
#     - Correlation Plot
#     - Dropping Features
#     - Label Encoding
#     - Missing value Imputation
#     - Normalization
#     - Train test Split
#     - Over-sampling
#     - PCA
# 
# #### 4. **Modelling**
#     - KNN Classifier
#     - Metrics Implementation
#     - ROC AUC Plot
# 
# #### 5. **Conclusion**
# 
# #### 6. **What's next ?**<br><br>
# 
# 
# ### Reason for Choosing this Dataset ?
# 
# - The Reason behind choosing this model is my Personal Interest to explore various Domains out there.
# 
# 
# - I want to investigate how Machine Learning can help to Forecast Weather from historical patterns. Where, ML can predict the likelihood of the Rainfall. Thereby, respective actions in the form of Treatments or Preventive Measures would be brought into consideration on the Individual.
# 
# 
# - However, this Statistical model is not prepared to use for production environment.
# 
# 
# ### Details :
# 
# This dataset contains daily weather observations from numerous Australia weather stations.
# The target variable RainTomorrow means: Did it rain the next day? Yes or No.
# 
# The description of data are as follows:
# - Date - Date of observation
# - Location - the common name of the location of the weather station
# - Min Temp - the minimum temperature in degree celsius
# - Max Temp - the maximum temperature in degree celsius
# - Rainfall - the amount of rainfall recorded for the day in mm
# - Evaporation - The so-called Class A pan evaporation (mm) in the 24 hours to 9am
# - Sunshine - The number of hours of bright sunshine in the day.
# - WindGustDir - The direction of the strongest wind gust in the 24 hours to midnight
# - WindGustSpeed - The speed (km/h) of the strongest wind gust in the 24 hours to midnight
# - WindDir9am - Direction of the wind at 9am
# - WindDir3pm - Direction of the wind at 3pm
# - WindSpeed9am - Wind speed (km/hr) averaged over 10 minutes prior to 9am
# - WindSpeed3pm - Wind speed (km/hr) averaged over 10 minutes prior to 3pm
# - Humidity9am - Humidity (percent) at 9am
# - Humidity3pm - Humidity (percent) at 3pm
# - Pressure9am - Atmospheric pressure (hpa) reduced to mean sea level at 9am
# - Pressure3pm - Atmospheric pressure (hpa) reduced to mean sea level at 3pm
# - Cloud9am - Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.
# - Cloud3pm - Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
# - Temp9am - Temperature (degrees C) at 9am
# - Temp3pm - Temperature (degrees C) at 3pm
# - RainToday - Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
# - RISK_MM - The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of measure of the "risk".
# - RainTomorrow - The target variable. Did it rain tomorrow?
# 
# 
# ### Questionnaire :
# 
# - Which cities do we have in our Dataset ?
# 
# 
# - Is our target Imbalanced ?
# 
# 
# - Can we extract Months / Years and plot the trend of Min. & Max. Temp for Canberra ? 
# 
# 
# - What is the relation of Rainfall & WindGust ? Do we see significant increase / decrease in WindGust with Rainfall / No Rainfall ?
# 
# 
# - Which cities experience immense rainfall (Avg. Rainfall) ?
# 
# 
# ### Objective
# 
# - Based on the given features, predict if it will rain tomorrow or not.
# 
# 
# - Only KNN Classifier must be used.

# ### Loading Data

# In[ ]:


#importing modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Reading the Dataset

df = pd.read_excel('WeatherAustralia.xlsx', header=0, parse_dates=True)
df.head()


# In[3]:


#Shape of dataset

print ('No. of Records :', df.shape[0], '\nNo. of Features : ', df.shape[1])


# - We can see that our Dataset contains 142193 examples and 24 features (including Target).

# In[4]:


#Let us check datset's attribute info

df.info()


# - Above results shows us that we've :
#     - **(17) x int64** datatype attributes
#     - **(6) x float64** datatype attributes
#     - **(1) x datetime64** datatype attribute

# ### Data Cleansing and EDA

# In[5]:


#Examining Null values in each feature

df.isnull().sum()


# - Above we can observe that we've lot of features with missing values.
# 
# 
# - Let us see the same in percentage.

# In[6]:


#%percent null values

df.isnull().sum() / df.shape[0] * 100


# - We can see that we've few features having missing values above 35%.
# 
# 
# - Usually in practice 20% to 25% missing value is accepted else we drop it unless we plan to use MICE (Multivariate Imputation by Chained Equations).

# In[7]:


#descriptive stats

df.describe()


# - Descriptive Stats give us better idea about range of continuous data.
# 
# 
# - We can observe how MinTemp's Minimum values is -8.5 degree celcius while Max. is 33.90 degree celcius. The values above 75% and below 25% can be considered as Outliers. We observe similar case with MaxTemp, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Temp9am, Temp3pm & RISK_MM.
# 
# 
# - Different variables have different scale. We neeed to Normalize the Dataset.

# In[8]:


#Brands Count

sns.set_context('talk')
plt.figure(figsize=(22,7))
sns.countplot(df['Location'], palette='Accent')
plt.xticks(rotation=90)
plt.title('Locations')


# - Above are the plot depicting count of records for Individual Location across Australia.
# 
# 
# - We've very less number of samples for Katherine, Uluru and Nhil. While, Canberra and Sydney claim the maximum samples.

# In[22]:


#Target Class count

plt.figure(figsize=(8,8))
plt.pie(df['RainTomorrow'].value_counts(), labels=['No','Yes'], autopct='%1.2f%%', explode=[0,0.2], shadow=True, colors=['crimson','gold'])
my_circle = plt.Circle( (0,0), 0.4, color='white')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Target Class Count')


# - Above plot tells us that we're dealing with Imbalanced dataset.
# 
# 
# - We'll need to over-sample the data set.

# In[10]:


#Time range

df['Date'].min(), df['Date'].max()


# - Our range od data dates from November'07 to July'17.
# 
# 
# - One thing to note that we do not have data of few beginning months of 2007 and ending months of 2017.
# 
# 
# - So we've close to 10 years of data at our hand.

# In[ ]:


#month and year extraction

df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year


# - Above we converting DateTime to Pandas DateTime and extracting Year and Months from it for further analysis.

# In[12]:


#min_max_temp_by_loc = 

canberra_df = df[df['Location'] == 'Canberra']

min_max_temp_by_loc = canberra_df.groupby(['Year','Month']).agg({'MinTemp':'mean',
                                                                 'MaxTemp':'mean'})


sns.set_context('paper')
plt.subplots_adjust(hspace=0.6)

plt.subplot(2,1,1)
min_max_temp_by_loc['MinTemp'].plot(figsize=(23,10), kind='bar')
plt.axhline(y=canberra_df['MinTemp'].mean(), color='red', linestyle='--')
plt.title('Canberra Min. Temp. Trend')
plt.plot()

plt.subplot(2,1,2)
min_max_temp_by_loc['MaxTemp'].plot(figsize=(23,8), kind='bar', color='g')
plt.axhline(y=canberra_df['MaxTemp'].mean(), color='red', linestyle='--')
plt.title('Canberra Max. Temp. Trend')
plt.plot()


# - Above we've plotted Canberra's Average Min. & Max. Temp over the period of decade. 
# 
# 
# - We are able to find out a definte trend in Avg. Max. Temp but difficult to find any similar pattern in Avg. Min. Temp. The trend is quite uneven.
# 
# 
# - One important thing we can infer is that May or June onwards there is always a significant drop in temperature and August onwards the temperature starts building up again significantly.

# In[13]:


#rainfall vs windgust 2016 - Canberra

canberra_monthly_avg_rainfall_windgust_2016 = canberra_df.groupby(['Year', 'Month']).agg({'Rainfall':'mean',
                                                                                          'WindGustSpeed':'mean'})

canberra_monthly_avg_rainfall_windgust_2016 = canberra_monthly_avg_rainfall_windgust_2016.reset_index()

canberra_monthly_avg_rainfall_windgust_2016 = canberra_monthly_avg_rainfall_windgust_2016[canberra_monthly_avg_rainfall_windgust_2016['Year'] == 2016].drop('Year', axis=1).set_index('Month')


#plotting rainfall & wingust
sns.set_context('talk')
canberra_monthly_avg_rainfall_windgust_2016.plot(secondary_y='WindGustSpeed', figsize=(10,5), marker='o')
plt.title('Canberra - Rainfall VS WindGust (2016)')


# - Above plot tells us that there is a correlation between Rainfall & WindGustSpeed.
# 
# 
# - As Rainfall (in MM) decreases then WindGust speed tend to decrease as well.
# 
# 
# - But thats not the case always, as we can observe Month of June, July, August & September. While in August Rainfall tend to decrease and windgust speed touches the peak of 48 (Secondary Y-axis).

# In[14]:


plt.figure(figsize=(8,6))
sns.boxenplot(data=canberra_df[canberra_df['Year'] == 2016], x = 'RainTomorrow', y='RISK_MM', #hue='RainTomorrow', 
              palette='seismic'
             )
plt.title('Risk MM by Rain Tomorrow')


# - Above plot shows Risk_MM vs Rain Tomorrow.
# 
# 
# - Risk MM states the amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of measure of the "risk".

# In[15]:


#location_wise_avg._rainfall_for_2016

location_wise_rainfall_2016 = df.groupby(['Year','Location']).agg({'Rainfall':'mean'})
location_wise_rainfall_2016 = location_wise_rainfall_2016.reset_index()
location_wise_rainfall_2016 = location_wise_rainfall_2016[location_wise_rainfall_2016['Year'] == 2016].drop('Year', axis=1).set_index('Location')
location_wise_rainfall_2016.sort_values(by='Rainfall', ascending=False).plot(kind='bar', figsize=(23,7), legend=False, colormap='seismic')
plt.title('Average Rainfall (in mm) by Location [2016]')
plt.ylabel('Avg. Rainfall (In mm)')
plt.xlabel('Location')


# - The above plot depicts Avg. rainfall of 2016 by Locations.
# 
# 
# - Its evident that Cairns has highest Avg. rainfall in 2016 which is almost 4.5 mm. While Woomera experienced Lowest rainfall in the year which is close to 1 mm.  

# In[16]:


#correlation

plt.figure(figsize=(20,10))
sns.heatmap(df.corr()*100, annot=True, cmap='winter')
plt.title('Correlation')


# - Pearson Correlation (Scaled to 0-100)
# 
# 
# - We can observe that there is a strong correlation between MinTemp & MaxTemp.
# 
# 
# - Also we can observe MinTemp, MaxTemp both have strong correlation with Temp9am & Temp3pm.
# 
# 
# - Pressure9am & Pressure3pm are strongly correlated too.
# 
# 
# - We can keep only 1 feature out of correlated ones but my plan is approach with Dimensionality reduction technique in Unsupervised called Principal Component Analysis (PCA). We do not need to rop features. 

# In[ ]:


#dropping columns with extensive missing values & date

df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Date'], axis=1, inplace=True)


# - We'll drop Date feature as we had extracted Year and Month.
# 
# 
# - Also, we'll drop features with extensive missing values.

# In[ ]:


#label encoding

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

df['WindGustDir'] = df['WindGustDir'].replace(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE','S', 'NW', 'SE', 'ESE', 'E', 'SSW'], 
                                              np.arange(0, len(pd.unique(df['WindGustDir']))-1))

df['WindDir9am'] = df['WindDir9am'].replace(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 'SSW', 'N','WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'],
                                            np.arange(0, len(pd.unique(df['WindDir9am']))-1))

df['WindDir3pm'] = df['WindDir3pm'].replace(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW', 'SW', 'SE', 'N', 'S', 'NNE', 'NE'],
                                            np.arange(0, len(pd.unique(df['WindDir3pm']))-1))

df['RainToday'] = df['RainToday'].replace(['No', 'Yes'], [0, 1])
df['RainTomorrow'] = df['RainTomorrow'].replace(['No', 'Yes'], [0, 1])
df['Location'] = enc.fit_transform(df['Location'])
df['Month'] = enc.fit_transform(df['Month'])
df['Year'] = enc.fit_transform(df['Year'])


# - We need to Label encoding to convert categorical features into numerical ones.
# 
# 
# - We are using Sklearn's LabelEncoder only to transform values of features with no missing values as it cannot deal with features having missing values, could throw error.
# 
# 
# - Instead we'll use replace feature of pandas to replace categorical features except NaN's. 

# ### Missing values Imputation

# In[19]:


#Let's fill NaN values now using FancyInput's IterativeImputer

from fancyimpute import IterativeImputer

df_filled = pd.DataFrame(data=IterativeImputer(imputation_order='ascending', n_iter=50, initial_strategy='median').fit_transform(df.values), columns=df.columns, index=df.index)
df_filled.head()


# - Above we had used IterativeImputer which is basedon Iterations.
# 
# 
# - We'll fill values going ascendingly for 50 Iterations having initial strategy as median because median values are not affected by outliers.
# 
# 
# - We can also use MICE for imputation but it needs system with big memory. It is a computational expensive technique. While using MICE we do not need to drop features. It can deal with even 50% of missing values.

# In[23]:


#rechecking null values

df_filled.isnull().any()


# In[24]:


# Normalising Predictors and creating new dataframe

from sklearn.preprocessing import StandardScaler

x = df_filled.drop('RainTomorrow', axis=1)
y = df_filled.RainTomorrow

cols = x.columns

ss = StandardScaler()

new_df = ss.fit_transform(x)
new_df = pd.DataFrame(new_df, columns=cols)
new_df.head()


# - Above we are Standardizing the data set using Standard Scaler.

# In[25]:


#importing train test split & some important metrics

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

x = new_df

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# - Above we're applying train test split on dataset before we perform over-sampling as our data set is highly imbalance. 

# ### Over-sampling using SMOTE

# In[ ]:


#implementing SMOTE

from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='not majority')
x_s_res, y_s_res = smote.fit_sample(x_train, y_train)

#print (y_train.value_counts(), '\n')
#np.bincount(y_s_res)


# - SMOTE works by creating synthetic observations based upon the existing minority observations (Chawla et al., 2002).
# 
# 
# -  For each minority class observation, SMOTE calculates the k nearest neighbors. Depending upon the amount of oversampling needed, one or more of the k-nearest neighbors are selected to create the synthetic examples.
# 
# 
# - ![alt text](https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/smote.png)

# ### PCA (Principal Components Analysis)

# - Why PCA ? 
# 
#     Ans : It's a Dimensionality Reduction Technique. It is also a Feature extraction Technique. By PCA we create new features from old (Original) Features but the new features will always be independent of each other. So, its not just Dimensionality Reduction Process, we are even eliminating Correlation between the Variables. As we have too many highly correlated features we are using PCA.

# In[27]:


#Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=x_s_res.shape[1])
pca.fit_transform(x_s_res)
var_rat = pca.explained_variance_ratio_
var_rat


# In[28]:


#Variance Ratio vs PC plot

plt.figure(figsize=(15,6))
plt.plot(np.arange(pca.n_components_), pca.explained_variance_, color="grey")
plt.title('Explained Variance by Number of Components')


# - At 11th component our PCA model seems to go Flat without explaining much of a Variance.
# 
# 
# - Component selection depends on person to person. It may not alwaysgive better results so we can try various numbers.
# 
# 
# - We'll proceed ahead with 11 components.

# In[ ]:


#Applying PCA as per required components

pca = PCA(n_components=11)
transform_train = pca.fit_transform(x_s_res)  #Transforming Train
transform_test = pca.transform(x_test)        #Transforming Validation Set


# ### KNN Classifier
# 
# 
# - Since our goal is to proceed with KNN Classifier we'll go with its implementation.
# 
# 
# - KNN Stores all available data and classifies on new input data based on Similarity.
# 
# 
# - The new instance is classified based on majority votes of its K-Neighbors. E.g.:, If given instance is close to Class 0 group than Class 1 group so the instance will belong to Class 0. 
# 
# 
# - The distance between the given instance and other data points is calculated using following measures :
# 
# 
# - ![alt text](https://www.saedsayad.com/images/KNN_similarity.png)
# 
# 
# - The lesser the distance between given instance and datapoints, the same class is assigned to our Instance based on Majority Voting.
# 
# 
# - ![alt_text](https://www.analyticsvidhya.com/wp-content/uploads/2014/10/scenario1.png)
# 
# 
# - Above Image states if blue star is our query then distance is computated to figure out if it belongs to Red Cluster / Class or Green Cluster / Class. 
# 
# 
# - ![alt_text](https://www.analyticsvidhya.com/wp-content/uploads/2014/10/scenario2.png)
# 
# 
# - We can observe that Post-distance calculation our algorithm figures out which class blue star belongs to and based on that the Red Class is assigned to our blue star.
# 
# 
# - Let us implement our KNN Classifier.

# In[ ]:


#Knn classifier

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(weights='uniform', metric='euclidean').fit(transform_train, y_s_res)
#knn_clf

pred = knn_clf.predict(transform_test)
pred


# In[ ]:


#Metrics evaluation

print (' Accuracy Score : ' ,accuracy_score(y_test, pred), '\n', 
       'Validation Score : ', knn_clf.score(transform_test, y_test), '\n',
       'Cross Validation Score : ', cross_val_score(knn_clf, transform_train, y_s_res, cv=5).mean(),'\n',
       'Classification Report : ', '\n', classification_report(y_test, pred))


# - Most important metrics from above is Recall.
# 
# 
# - Accuracy can be a illusion in Imbalanced dataset.
# 
# 
# - As we had obtained better recall score, we can say that our model is better at classifying both classes.

# In[ ]:


#Confusion matrix

sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='seismic')
plt.title('Confusion Matrix')


# - Observation tells us that our model is acceptable. We had classified 23367 records correctly as it isn't going to rain. While, 6226 records classified as it will rain.
# 
# 
# - Interpretation of Confusion Matrix is as follows :
# 
# 
# ![alt_text](https://miro.medium.com/max/356/1*Z54JgbS4DUwWSknhDCvNTQ.png)

# ### ROC AUC
# 
# 
# - When it comes to a classification problem, we can count on an AUC - ROC Curve. When we need to check or visualize the performance of the multi - class classification problem, we use AUC (Area Under The Curve) ROC (Receiver Operating Characteristics) curve. 
# 
# 
# - It is one of the most important evaluation metrics for checking any classification model’s performance.

# In[ ]:


#Roc-Auc

from sklearn.metrics import auc, roc_auc_score, roc_curve

prob = knn_clf.predict_proba(transform_test)

fpr, tpr, _ = roc_curve(y_test, prob[:,1])

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,7))

plt.plot(fpr, tpr, color=(np.random.rand(), np.random.rand(), np.random.rand()), label='AUC = %0.4f'% roc_auc)
plt.plot([0,1], 'grey', lw=2, linestyle='-.')

plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver operating characteristic - Area Under Curve (ROC-AUC)')


# - Above we had achieved AUC of 87.97% (~88%). Which is better.
# 
# 
# - Performance of model is called excellent when AUC is close to 1.
# 
# 
# - While, close to 0 is termed as worst performing model.

# ### Conclusion :
# 
# 
# - We figured out how our dataset was suffering from Class imbalance & so We handled imbalanced dataset with the help of SMOTE.
# 
# 
# - We had achieved feat of 0.8797 AUC which is acceptable. 

# ### What's next ?
# 
# 
# - We can also try to **add more Parameters** for **Tuning the model.**
# 
# 
# - One can also try various ways to impute Missing Values.
# 
# 
# - One can try also **implement Ensemble method, Tree Algorithm and / or Deep Neural Network** Modelling.
