import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


data = pd.read_csv('cleaned_cuisines.csv')

X = data.iloc[:,2:]

y = data[['cuisine']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

model = SVC(kernel='linear', C=10, probability=True,random_state=0)
model.fit(X_train,y_train.values.ravel())

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))

initial_type = [('float_input', FloatTensorType([None, 380]))]
options = {id(model): {'nocl': True, 'zipmap': False}}

onx = convert_sklearn(model, initial_types=initial_type, options=options)
with open("./model.onnx", "wb") as f:
    f.write(onx.SerializeToString())