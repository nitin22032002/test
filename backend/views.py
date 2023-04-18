from django.http import JsonResponse
from backend.settings import BASE_DIR
from django.views.decorators.csrf import csrf_exempt
import pickle
import numpy as np
import json
model=pickle.load(open(f"{BASE_DIR}/ML/model/model.pkl","rb"))

@csrf_exempt
def fetchAppointment(request):
    try:
        data=json.loads(request.body)
        fields=['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']
        values=[]
        for item in fields:
            if(item not in data):
                return JsonResponse({"status":False,"err":f"{item} Field Missing"})
            values.append(data[item])
        values=np.array(values).reshape((1,-1))
        output=model.predict_proba(values)
        return JsonResponse({"status":True,"prob-neg":output[0][0],"prob-pos":output[0][1]})
    except Exception as e:
        print(e)
        return JsonResponse({"status":False})