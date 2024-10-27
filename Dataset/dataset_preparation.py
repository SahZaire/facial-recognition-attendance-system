import numpy as np
import os
import random
import pandas as pd

people_biodata = pd.read_csv("Minor Project/Facial Recognition Attendance Sysytem/final_dataset/TY CS/listed_100_biodata.csv")

people_biodata["Roll_num"] = ["B{}" .format(i+1) for i in range(len(people_biodata))]
people_biodata["Reg_num"] = ["22BCSE{}" .format(random.randint(1000000, 9999999)) for _ in range(len(people_biodata))]

subjects = ['DAA', 'SEPM', 'DL', 'NLP', 'CN', 'APT', 'EEIM', 'MP', 'DAA_PR', 'DL_PR', 'NLP_PR']
for subject in subjects:
    people_biodata[subject] = 0

print(people_biodata.head())
people_biodata.to_csv("Minor Project/Facial Recognition Attendance Sysytem/final_dataset/TY CS/listed_100_biodata.csv", index=False)
