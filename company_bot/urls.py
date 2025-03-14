from django.contrib import admin
from django.urls import path, include

API = 'api/v0'

urlpatterns = [
    path('/duisbxtya/admin/', admin.site.urls),
    path(f'{API}/chats/', include('chats.urls')),
]
