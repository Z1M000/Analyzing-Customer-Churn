import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



data=pd.read_csv("telecom_churn.csv")

print(data.isnull().sum()) #check if anything is null 

#CustServCalls vs Churn
plt.figure()
sns.boxplot(x="Churn",y="CustServCalls",data=data)
plt.title("Customer Service Calls vs Churn")
plt.show()

#MonthlyCharge vs Churn
plt.figure()
sns.boxplot(x="Churn",y="MonthlyCharge",data=data)
plt.title("Monthly Charge vs Churn")
plt.show()

#DataUsage vs Churn
plt.figure()
sns.boxplot(x="Churn",y="DataUsage",data=data)
plt.title("Data Usage vs Churn")
plt.show()

#check correlations between attributes 
corr=data.drop(columns=["Churn"]).corr() 
print(corr)


clean=data.drop(columns=["DataPlan"]) #we dropped DataPlan since it was highly correlated with another attribute

summary=clean[["AccountWeeks","DataUsage", "CustServCalls","DayMins", "DayCalls", "MonthlyCharge","OverageFee", "RoamMins"]].describe()

print(summary) #plot summary stats of non binary features 

#scale the data:
scaled=clean.copy()
features=scaled.drop(columns=["Churn"]) #dont include label in scaling 
scaler=StandardScaler()
scaled_features=scaler.fit_transform(features)
scaled[features.columns]=scaled_features

#save csv
scaled.to_csv("cleaned_dataset.csv", index=False)