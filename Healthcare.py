# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:02:29 2026

@author: syeda
"""

import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\Users\syeda\OneDrive\Desktop\Healthcare_Analysis\healthcare_dataset.csv")

print(df.head(4))
print(df.dtypes)
print(df.shape)
print(df.isna().sum()) #no null
print(df.columns)


#Droping Uselss cols
df=df.drop(["Name","Doctor","Hospital","Room Number"],axis=1)
print(df.columns)

#Onehot 
df=pd.get_dummies(df,columns=["Blood Type","Gender","Medical Condition",
                              "Insurance Provider","Admission Type",
                              "Medication"
                              ],dtype=int,drop_first=True)


df["Test Results"] = df["Test Results"].map({
    "Abnormal": 0,
    "Normal": 1,
    "Inconclusive": 2
})

print(df.dtypes)

#Feature Engineering
print(df[["Date of Admission", "Discharge Date"]].head(10))

df["Date of Admission"]=pd.to_datetime(df["Date of Admission"],dayfirst=True)
df["Discharge Date"]=pd.to_datetime(df["Discharge Date"],dayfirst=True)

df["Doa Year"] = df["Date of Admission"].dt.year
df["Doa Month"] = df["Date of Admission"].dt.month 
df["Doa Day"] = df["Date of Admission"].dt.day 
df["DYear"] = df["Discharge Date"].dt.year 
df["DMonth"] = df["Discharge Date"].dt.month 
df["Dday"] = df["Discharge Date"].dt.day

print((df["Date of Admission"] == df["Discharge Date"]).value_counts())

df["Length Of Stay"] = (
    df["Discharge Date"] - df["Date of Admission"]
).dt.days

print(df["Length Of Stay"].head(10))
print(df["Length Of Stay"].describe())
print(df["Length Of Stay"].unique()[:20])
df = df.drop(columns=["Date of Admission","Discharge Date"])

df.to_csv(
    r"C:\Users\syeda\OneDrive\Desktop\Healthcare_Analysis\Healthcare_Model_Ready.csv",
    index=False
)

print("Model-ready dataset saved successfully!")