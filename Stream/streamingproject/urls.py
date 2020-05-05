
from django.conf.urls import url
from django.contrib import admin
from django.urls import path

from streamingproject import views

# url patterns for browsing 

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    # При переходе на stream/1/1.mp4 будет обрабатываться одно видео
    # При переходе на stream/2/2.mp4 будет обрабатываться другое видео
    url(r'^stream/(?P<num>\d+)/(?P<stream_path>(.*?))/$',views.dynamic_stream,name="dynamic_stream"),
    url(r'^index/$', views.indexscreen),
    url(r'^menu/$', views.menu),
    url(r'^logout/',views.log_out),
]
