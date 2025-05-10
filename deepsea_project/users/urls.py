from django.urls import path
from django.contrib.auth.views import LogoutView
from . import views

app_name = 'users'

urlpatterns = [
    path('getstarted/', views.getstarted, name='getstarted'),
    path('logout/', views.logout_view, name='logout'),  # <- Add this
]
