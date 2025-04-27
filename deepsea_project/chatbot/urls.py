from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chatbot/', views.chatbot_page, name='chatbot'),
    path('get-response/', views.get_bot_response, name='get_bot_response'),

]
