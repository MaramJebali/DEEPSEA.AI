from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chatbot/', views.chatbot_page, name='chatbot'),
    path('aqua/', views.aqua, name='aqua'),
    path('overfish/', views.overfish, name='overfish'),


    path('Marinehealth/', views.marinehealth, name='Marinehealth'),
    path('get-response/', views.get_bot_response, name='get_bot_response'),
    path('noise-analysis/', views.noise_analysis, name='noise_analysis'),

]
