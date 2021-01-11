from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('statistics/', views.statistics, name='statistics'),
    path('<str:image_id>/', views.detail, name='detail'),
    path('<str:image_id>/results/', views.results, name='results'),
    path('<str:image_id>/tag/', views.tag, name='tag'),
]
