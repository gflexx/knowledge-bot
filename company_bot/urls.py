from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from django.conf import settings



API = 'api/v0'

urlpatterns = [
    path('/duisbxtya/admin/', admin.site.urls),
    path(f'{API}/chats/', include('chats.urls')),
]

urlpatterns = urlpatterns + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns = urlpatterns + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)