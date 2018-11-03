from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.core import serializers
from django.utils import timezone
from django import forms
from django.template import RequestContext
from django.contrib import auth
from django.contrib.auth.models import User

# Create your views here.
"""
Index page
"""
def index(request):
    return render(request, 'index.html', {})

"""
User Registraion Process
"""
def register(request):
    if request.method == "POST":
        user_form = UserFormRegister(request.POST)
        if user_form.is_valid():
            # Get info from form
            username = user_form.cleaned_data['username']
            # pdb.set_trace()
            filter_results = User.objects.filter(username=username)
            if len(filter_results) > 0:
                # Generate context
                context = {
                    'error_msg' : "User already exist!",
                }
                return render(request, 'user_register.html', context)
            else:
                password = user_form.cleaned_data['password']
                password_confirm = user_form.cleaned_data['password_confirm']
                if password != password_confirm:
                    # Generate context
                    context = {
                        'error_msg' : "Password not consistent!",
                    }
                    return render(request, 'user_register.html', context)
                # Save form content into database
                User.objects.create_user(username=username, password=password)
                return HttpResponseRedirect('/index/')
    else:
        return render(request, 'user_register.html', {})

"""
User Login Process
"""
def login(request):
    if request.method == "POST":
        user_form = UserFormLogin(request.POST)
        if user_form.is_valid():
            # Get info from the form
            username = user_form.cleaned_data['username']
            password = user_form.cleaned_data['password']
            user = auth.authenticate(username=username, password=password)
            # pdb.set_trace()
            if user:
                auth.login(request, user)
                return HttpResponseRedirect('/index/')
            else:
                context = {
                    'error_msg' : 'User does not exist!',
                }
                return render(request, "user_login.html", context)
    else:
        return render(request, "user_login.html", {})

"""
User Logout Process
"""
def logout(request):
    if request.method == 'GET':
        auth.logout(request)
        return HttpResponseRedirect('/login/')

class UserFormRegister(forms.Form):
    username = forms.CharField(label='Username', max_length=100)
    password = forms.CharField(label='Password', widget=forms.PasswordInput())
    password_confirm = forms.CharField(label='Confirm Password', widget=forms.PasswordInput())

class UserFormLogin(forms.Form):
    username = forms.CharField(label='Username',max_length=100)
    password = forms.CharField(label='Password',widget=forms.PasswordInput())
