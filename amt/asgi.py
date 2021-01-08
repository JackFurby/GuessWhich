import os
from channels.asgi.urls import get_channel_layer
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demo.settings")
django_asgi_app = get_asgi_application()

channel_layer = get_channel_layer()

#application = ProtocolTypeRouter({
#    "websocket": urls.urlpatterns
#})
