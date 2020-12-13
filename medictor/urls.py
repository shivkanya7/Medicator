"""medictor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from . import views

urlpatterns = [

    path("", views.home, name="home"),
    path('about/',views.about,name='about'),
    path('sign_up_patient/',views.sign_up_patient,name='sign_up_patient'),
    path('user_profile_patient/',views.user_profile_patient,name='user_profile_patient'),
    path('sign_in_patient/',views.sign_in_patient,name='sign_in_patient'),
    path('diseasepred/',views.diseasepred,name="diseasepred"),
    path('logout_patient/',views.logout_patient,name="logout_patient"),
    path('input_symptoms/',views.input_symptoms,name="input_symptoms")
]
