import json
import requests

def predict_diabetes(BMI,Age,Glucose):
    url='http://127.0.0.1:5000/diabetes/v1/predict'
    data={"BMI":BMI,"Age":Age,"Glucose":Glucose}
    data_json=json.dumps(data)
    headers={'content-type':'application/json'}
    response=requests.post(url,data=data_json,headers=headers)
    result=json.loads(response.text)
    return result

def predict_heartdisease(age,chol,thalach):
    url='http://127.0.0.1:5001/heartDisease/v1/predict'
    data={"Age":age,"Cholesterol":chol,"Max HeartRate":thalach}
    data_json=json.dumps(data)
    headers={'content-type':'application/json'}
    response=requests.post(url,data=data_json,headers=headers)
    result=json.loads(response.text)
    return result

if __name__=="__main__":
    
    BMI=int(input('BMI -- '))
    Age=int(input('Age -- '))
    Glucose=int(input('Glucose -- '))
#    age=int(input('Enter age again -- '))
#    chol=int(input('Cholesterol -- '))
#    thalach=int(input('Max HeartRate -- '))
    
    predictions=predict_diabetes(BMI,Age,Glucose)
#    predictions1=predict_heartdisease(age,chol,thalach)
    
    print("Diabetic" if predictions["prediction"]==1 else "Not Diabetic")
    print("Confidence -- "+predictions["confidence"]+"%")
    
#    print("Heart Disease" if predictions1["prediction"]==1 else "No Heart Disease")
#    print("Confidence -- "+predictions1["confidence"]+"%")
