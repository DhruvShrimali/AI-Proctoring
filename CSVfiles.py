import pandas as pd
df = pd.read_csv('8.csv') #Change file number here and last line
for i in range(1, 300):
        name = "HR_"+str(i)
        num = i*(-1)
        df[name] = df["H_Roll"].shift(num)
for i in range(1, 300):
        name = "HP_"+str(i)
        num = i*(-1)
        df[name] = df[" H_Pitch"].shift(num)
for i in range(1, 300):
        name = "HY_"+str(i)
        num = i*(-1)
        df[name] = df[" H_Yaw"].shift(num)
for i in range(1, 300):
        name = "EP_"+str(i)
        num = i*(-1)
        df[name] = df[" E_Pitch"].shift(num)
for i in range(1, 300):
    name = "EY_"+str(i)
    num = i*(-1)
    df[name] = df[" E_Yaw"].shift(num)
for i in range(1, 300):
    name = "C_"+str(i)
    num = i*(-1)
    df[name] = df[" Cheat"].shift(num)
ORS = df[" Cheat"] + df["C_1"]
for i in range(3, 300):
    name1 = "C_" + str(i)
    name2 = "C_" + str(i-1)
    ORS = df[name1] + df[name2]
df["OC"]=ORS
df["Cheating"] = (df["OC"] > 1).astype(int)
del df["OC"]
del df[" Cheat"]
for i in range(1, 300):
        name = "C_"+str(i)
        del df[name]
df = df.dropna()

df.to_csv('file8.csv', header=False, index=False) #header = True for first csv file compiled