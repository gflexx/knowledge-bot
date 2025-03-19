from django.urls import path
from .views import *

urlpatterns = [
    path('answer/', answer_question, name='answer_question'),
    path('list/create/', DocumentListCreateAPiView.as_view()),

]