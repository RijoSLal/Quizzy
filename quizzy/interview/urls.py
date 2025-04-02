"""
URL configuration for quizzy project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from .views import Myview ,Interview,Cam,PredictionView,Home,Check,Score,No_Stream_Cam
urlpatterns = [
    path("",Home.as_view(),name="home"),
    path("eligibility/",Myview.as_view(),name="eligibility"),
    path("interview/",Interview.as_view(),name="interview"),
    path("camera/",Cam.as_view(),name="live"),
    path("lite/",No_Stream_Cam.as_view(),name="lite"),
    path('prediction/', PredictionView.as_view(), name='prediction'),
    path("time/",Check.as_view(),name="time"),
    path("score/",Score.as_view(),name="score"),
    
]
