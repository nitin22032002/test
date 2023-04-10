from django.http import JsonResponse
import pickle
from django.views.decorators.csrf import csrf_exempt
import json
from backend.settings import BASE_DIR
import numpy as np

model=pickle.load(open(f"{BASE_DIR}/Model/model.pkl","rb"))

def detect(data):
    columns=['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'ca']
    values={}
    for item in columns:
        values[item]=float(data[item])
    values['cp_0']=values['cp_1']=values['cp_2']=values['cp_3']=values['thal_0']=values['thal_1']=values['thal_2']=values['thal_3']=values['slope_0']=values['slope_1']=values['slope_2']=0
    cp=data['cp']
    thal=data['thal']
    slope=data['slope']
    values[f'cp_{cp}']=values[f'cp_{thal}']=values[f'cp_{slope}']=1
    return values

@csrf_exempt
def findHeart(request):
    try:
        data=json.loads(request.body)
        values=detect(data)
        values=np.array(list(values.values()))
        values=np.reshape(values,(1,-1))
        output=model.predict_proba(values)
        return JsonResponse({"status":True,"prob":output[0][0]})
    except Exception as e:
        print(e)
        return JsonResponse({"status":False,"err":"Server Error..."},status=500)