
from django.contrib import admin
from django.urls import path
from .views import predictPrice
urlpatterns = [
    path("have",predictPrice)
]
