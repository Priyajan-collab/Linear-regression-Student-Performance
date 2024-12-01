import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

student_data=pd.read_csv("data/Student_Performance.csv")
student_data["Extracurricular Activities"]=student_data["Extracurricular Activities"].map({"Yes":1,"No":0})
# splitting data as of now there is only training set, as I am only learning to apply linear regression
y=student_data["Performance Index"]
x=student_data.drop(["Performance Index"],axis="columns")
# converting them to numpy array
x=x.to_numpy()
y=y.to_numpy()


x=(x-np.mean(x,axis=0))/np.std(x)
y=(y-np.mean(y,axis=0))/np.std(y)


# setting up graphs
# fig,ax= pt.subplots(1,5,sharey=True,figsize=(12,5))
fig,ax=pt.subplots()

def show_data():
    for i in range(len(ax)):
        ax[i].set_ylim(-1,1)
        ax[i].set_xlim(-1,1)
        ax[i].plot(x[:,i],y,c="r")
        ax[i].set_ylabel("Performance Index")
        ax[i].legend()
# show_data()

# setting up weights and biases
b=np.random.uniform(0,1)
w=np.zeros(x.shape[1])
print(w)

class Linear_Regression():
    pass

pt.show()

