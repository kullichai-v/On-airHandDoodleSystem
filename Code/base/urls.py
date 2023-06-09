from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register, name="register"),
    path("login/", views.login, name="login"),
    path("forgot/", views.forgot, name="forgot"),
    path("profile/", views.profile, name="profile"),
    path("password/", views.password, name="password"),
]
