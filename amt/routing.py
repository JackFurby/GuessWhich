#from channels.routing import route, include
#from .consumers import ws_message, ws_connect, ws_disconnect

#ws_routing = [
#    route("websocket.receive", ws_message),
#    route("websocket.connect", ws_connect),
#]

#channel_routing = [
#    include(ws_routing, path=r"^/chat"),
#]

from django.core.asgi import get_asgi_application
from django.conf.urls import url
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

from .consumers import ChatConsumer

channel_routing = ProtocolTypeRouter({
    # Django's ASGI application to handle traditional HTTP requests
    "http": get_asgi_application(),

    # WebSocket chat handler
    "websocket": AuthMiddlewareStack(
        URLRouter([
            url(r"^chat/$", ChatConsumer.as_asgi()),
        ])
    ),
})
