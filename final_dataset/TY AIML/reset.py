import pandas as pd

df = pd.read_csv("Minor Project/Facial Recognition Attendance Sysytem/final_dataset/TY AIML/listed_100_biodata.csv")

SUB = "ex_atn"
df = df.drop(SUB, axis =1)
df["DAA"] = 0
df["SEPM"] = 0
df["DL"] = 0
df["NLP"] = 0
df["CN"] = 0
df["APT"] = 0
df["EEIM"] = 0
df["MP"] = 0
df["DAA_PR"] = 0
df["DL_PR"] = 0
df["NLP_PR"] = 0

df.to_csv("Minor Project/Facial Recognition Attendance Sysytem/final_dataset/TY AIML/listed_100_biodata.csv", index=False)