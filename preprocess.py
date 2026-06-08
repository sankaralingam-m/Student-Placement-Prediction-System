import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib

df=pd.read_csv("data/train.csv")
df=df.drop("Student_ID",axis=1)

le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
df["Degree"]=le.fit_transform(df["Degree"])
df["Branch"]=le.fit_transform(df["Branch"])
df["Placement_Status"]=le.fit_transform(df["Placement_Status"])


x=df.drop("Placement_Status",axis=1)
y=df["Placement_Status"]
print("x=",x.shape)
print("y=",y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)



cm = confusion_matrix(y_test, y_pred)




joblib.dump(model, "model/placement_model.pkl")

print("Model Saved Successfully")