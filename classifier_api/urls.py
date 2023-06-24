from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import classifyImageView

urlpatterns = [
    path('classify/', classifyImageView.as_view()),
]
