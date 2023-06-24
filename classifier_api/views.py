from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response

from .utils import load_prepare_image_torch, create_torch_model, model_pred_torch
from .serializers import ImageClassificationResultSerializer
from .models import ImageClassificationModel 

import os

class classifyImageView(APIView):
    def post(self, request):
        boolImage = 'image' in request.FILES
        if request.method == 'POST' and 'image' in request.FILES:
            image_file = request.FILES.get('image')
            
            # loading model
            model_path = os.path.join(os.path.dirname(__file__), 'modelWeights', 'seeFoodModel_rexnet_2.pt')
            
            model, transform = create_torch_model(model_path)
            input_image = load_prepare_image_torch(image_file, transform)
            food_name = model_pred_torch(model, input_image)
            
            result = ImageClassificationModel(image=image_file, classifiedLabel=food_name)
            result.save()
            serializer = ImageClassificationResultSerializer(result)
            
            return_json = {
                'foodName': food_name,
                'status': 200,
            }
            
            return JsonResponse(return_json)
        
        error_response = {'error': 'Invalid request', 'methon': request.method, 'image is there': boolImage}
        return JsonResponse(error_response, status=400)