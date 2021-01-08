import os
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import amt.routing

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demo.settings")

application = ProtocolTypeRouter({
    # Django's ASGI application to handle traditional HTTP requests
    "http": get_asgi_application(),
    # WebSocket chat handler
    "websocket": AuthMiddlewareStack(
        URLRouter(
            amt.routing.websocket_urlpatterns
        )
    ),
})
