# urls.py
from django.urls import path
from . import views  # ou from outcomes import views selon ton architecture

urlpatterns = [
    path('coral/', views.coral_classifier_view, name='coral_classifier'),
    path('zoo/', views.zooplankton_classifier_view, name='zoo_classifier'),

]
