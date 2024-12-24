from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('results/<int:video_id>/', views.results, name='results'),
    path('fetch_previous_results/', views.fetch_previous_results, name='fetch_previous_results'),
]