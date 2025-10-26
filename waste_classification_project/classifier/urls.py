from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('classify/', views.classify_image, name='classify'),
    path('result/', views.result, name='result'),
]