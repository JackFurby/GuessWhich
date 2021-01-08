from django.conf.urls import url, include
from django.urls import path
from django.contrib import admin

urlpatterns = [
    url('admin', admin.site.urls),
    url('', include('amt.urls')),
]
#] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
