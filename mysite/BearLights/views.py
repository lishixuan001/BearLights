from __future__ import print_function
import json
from os.path import join, dirname
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback, AudioSource
import threading
import os
import scipy.io as spio

from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.core import serializers
from django.utils import timezone
from django import forms
from django.template import RequestContext
from django.contrib import auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required



from BearLights.models import *

"""
Index page
"""
def index(request):
    context = {}

    if request.user.is_authenticated:
        return HttpResponseRedirect("/profile/")

    if request.method == "POST":
        audio_form = IndexForm(request.POST)
        if audio_form.is_valid():
            audio_string = audio_form.cleaned_data['data']

            file_path = "sound.wav"

            wav_file = open(file_path, 'wb')
            audio_data = audio_string.strip().split(",")
            for num in audio_data:
                num = int(num)
                num_byte = num.to_bytes((num.bit_length() + 7) // 8, byteorder='big')
                wav_file.write(num_byte)
            wav_file.close()

            read_file = open(file_path, 'rb')
            print(read_file.readlines())

        # If service instance provides API key authentication
        service = SpeechToTextV1 (
            ## url is optional, and defaults to the URL below. Use the correct URL for your region.
            url='https://gateway-wdc.watsonplatform.net/speech-to-text/api',
            iam_apikey='iI6HjiOk8o2mMa86Nic4cSgAF9Sqhhp0bIoGcGo1BT63'
        )

        with open(file_path, 'rb') as audio_file:
            result = service.recognize (
                        audio=audio_file,
                        content_type='audio/wav',
                        timestamps=True,
                        word_confidence=True
                        ).get_result()
            text = result["results"][0]["alternatives"][0]["transcript"]

            print(text)

        os.remove(file_path)

        # Check if page jumping is needed
        if "log in" in text:
            return HttpResponseRedirect('/accounts/login/')
        elif "register" in text:
            return HttpResponseRedirect('/accounts/register/')
        else:
            err_msg = "Sorry, we did not understand your command."
            context.update({"err_msg": err_msg})
            render(request, 'index.html', context)

    return render(request, 'index.html', context)

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
        return HttpResponseRedirect('/index/')

"""
Profile page
"""
@login_required
def profile(request):
    current_user = request.user
    username = current_user.get_username()
    context = {
        'username' : username,
    }
    return render(request, "user_profile.html", context)

"""
Form Helper Methods
"""
class IndexForm(forms.Form):
    data = forms.CharField(label="Data", max_length=None)

class UserFormRegister(forms.Form):
    username = forms.CharField(label='Username', max_length=100)
    password = forms.CharField(label='Password', widget=forms.PasswordInput())
    password_confirm = forms.CharField(label='Confirm Password', widget=forms.PasswordInput())

class UserFormLogin(forms.Form):
    username = forms.CharField(label='Username',max_length=100)
    password = forms.CharField(label='Password',widget=forms.PasswordInput())
