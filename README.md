# Australia - Rain-Prediction

### Table of Contents

#### 1. **Information**
    - Reason for Choosing this Dataset ?**
    - Details
    - Objective

#### 2. **Loading Dataset**
    - Importing packages
    - Reading Data
    - Shape of data
    - Dtype

#### 3. **Data Cleansing & EDA**
    - Checking Null values
    - Descriptive Statistics
    - Viz. (Phase 1)
    - Month & Year Extraction
    - Viz. (Phase 2)
    - Correlation Plot
    - Dropping Features
    - Label Encoding
    - Missing value Imputation
    - Normalization
    - Train test Split
    - Over-sampling
    - PCA

#### 4. **Modelling**
    - KNN Classifier
    - Metrics Implementation
    - ROC AUC Plot

#### 5. **Conclusion**

#### 6. **What's next ?**<br><br>


### Reason for Choosing this Dataset ?

- The Reason behind choosing this model is my Personal Interest to explore various Domains out there.


- I want to investigate how Machine Learning can help to Forecast Weather from historical patterns. Where, ML can predict the likelihood of the Rainfall. Thereby, respective actions in the form of Treatments or Preventive Measures would be brought into consideration on the Individual.


- However, this Statistical model is not prepared to use for production environment.


### Details :

This dataset contains daily weather observations from numerous Australia weather stations.
The target variable RainTomorrow means: Did it rain the next day? Yes or No.

The description of data are as follows:
- Date - Date of observation
- Location - the common name of the location of the weather station
- Min Temp - the minimum temperature in degree celsius
- Max Temp - the maximum temperature in degree celsius
- Rainfall - the amount of rainfall recorded for the day in mm
- Evaporation - The so-called Class A pan evaporation (mm) in the 24 hours to 9am
- Sunshine - The number of hours of bright sunshine in the day.
- WindGustDir - The direction of the strongest wind gust in the 24 hours to midnight
- WindGustSpeed - The speed (km/h) of the strongest wind gust in the 24 hours to midnight
- WindDir9am - Direction of the wind at 9am
- WindDir3pm - Direction of the wind at 3pm
- WindSpeed9am - Wind speed (km/hr) averaged over 10 minutes prior to 9am
- WindSpeed3pm - Wind speed (km/hr) averaged over 10 minutes prior to 3pm
- Humidity9am - Humidity (percent) at 9am
- Humidity3pm - Humidity (percent) at 3pm
- Pressure9am - Atmospheric pressure (hpa) reduced to mean sea level at 9am
- Pressure3pm - Atmospheric pressure (hpa) reduced to mean sea level at 3pm
- Cloud9am - Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.
- Cloud3pm - Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
- Temp9am - Temperature (degrees C) at 9am
- Temp3pm - Temperature (degrees C) at 3pm
- RainToday - Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
- RISK_MM - The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of measure of the "risk".
- RainTomorrow - The target variable. Did it rain tomorrow?


### Objective

- Based on the given features, predict if it will rain tomorrow or not.


- Only KNN Classifier must be used.
