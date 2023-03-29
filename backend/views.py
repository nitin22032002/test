from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
from backend.settings import BASE_DIR
import numpy as np

@csrf_exempt
def predictPrice(request):
    try:
        data=json.loads(request.body)
        params="Area,No. of Bedrooms,Gymnasium,SwimmingPool,LandscapedGardens,IndoorGames,SportsFacility,ATM,ClubHouse,24X7Security,PowerBackup,CarParking,LiftAvailable,furnishing".split(",")
        values=[]
        for item in params:
            if(item not in data):
                return JsonResponse({"status":False,"msg":f"Parameter {item} is missing"},status=400)
            values.append(data[item])
        model=pickle.load(open(f"{BASE_DIR}/ML/Model/model.pkl","rb"))
        values=np.array(values)
        values=np.reshape(values,(1,-1))
        output=model.predict(values)[0]*(10**6)
        return JsonResponse({"status":True,"price":output})
    except Exception as e:
        print(e)
        return JsonResponse({"Status":False,"msg":str(e)},status=500)