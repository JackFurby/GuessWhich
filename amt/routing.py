from django.urls import re_path
from django.conf.urls import url

from . import consumers

websocket_urlpatterns = [
    #re_path(r'ws/chat/$', consumers.ChatConsumer.as_asgi()),
    #url(r"^chat/$", consumers.ChatConsumer.as_asgi()),
    re_path(r'chat/$', consumers.ChatConsumer.as_asgi()),
    #re_path(r'chat/(?P<socketid>\w+)/$', consumers.ChatConsumer.as_asgi()),
]
