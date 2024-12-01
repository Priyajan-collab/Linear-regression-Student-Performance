import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

student_data=pd.read_csv("data/Student_Performance.csv")
student_data["Extracurricular Activities"]=student_data["Extracurricular Activities"].map({"Yes":1,"No":0})
# splitting data as of now there is only training set, as I am only learning to apply linear regression
y=student_data["Performance Index"]
x=student_data.drop(["Performance Index"],axis=1)
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
lr=0.0025
print(x.shape,w.shape)
class Linear_Regression():
    def __init__(self,x,y,w,b,lr):
        self.x=x
        self.y=y
        self.w=w
        self.b=b
        self.lr=lr
        self.len=len(y)
    def predict(self):
        self.ys=np.dot(self.x,self.w)+self.b
        return self.ys
     
    def cost(self):
        self.diff=self.y-self.ys
        self.error=np.sum(self.diff**2)/(2*self.len)
        # print(self.error)
    def update_weights_biases(self):
        d_w = np.dot(self.x.T, self.diff) / self.len
        d_b=np.sum(self.diff)/self.len
        self.w+=self.lr*d_w;
        self.b+=self.lr*d_b
    
    def r2_score(self):
        total_sum_of_squares = np.sum((self.y - np.mean(self.y))**2)
        residual_sum_of_squares = np.sum((self.y - self.ys)**2)
        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r2
        
obj=Linear_Regression(x,y,w,b,lr)
x=[]
y=[]

for i in range(1000):
    obj.predict()
    obj.cost()
    obj.update_weights_biases()

    print("accuracy is" ,obj.r2_score())
    y.append(obj.error)
    x.append(i)
new_w=obj.w
new_b=obj.b
# print(new_w,new_b)
predict=np.array([8	,51	,1,	7	,2])
predict = (predict - np.mean(student_data.drop(["Performance Index"],axis=1).to_numpy(), axis=0)) / np.std(student_data.drop(["Performance Index"],axis=1).to_numpy(), axis=0)


# print(predict.shape)
ax.plot(x,y,c="r")
predicted_value=np.dot(predict,new_w )+new_b
predicted_value = predicted_value * np.std(student_data["Performance Index"]) + np.mean(student_data["Performance Index"])

# print("required value",y[0])

print("the predicted value for",predicted_value)

pt.show()



