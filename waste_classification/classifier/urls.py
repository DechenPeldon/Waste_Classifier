from django.urls import path
from . import views

app_name = 'classifier'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('classify/', views.classify_waste, name='classify'),
    path('result/<int:result_id>/', views.result, name='result'),
]