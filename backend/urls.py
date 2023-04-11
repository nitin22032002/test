
from django.contrib import admin
from django.urls import path
from .views import predictPrice
# from ML.MLModel import program,train_model
urlpatterns = [
    path("have",predictPrice)
]
