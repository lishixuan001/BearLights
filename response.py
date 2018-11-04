import speech_recognition as sr
from gtts import gTTS
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

from data import *
from random import randint
""""""
# STYLE = None
def response(audio, step, style_name, color_check = False):
    # global STYLE
    STYLE = style_name
    if step == 0:
        welcometext = 'Hi, what kind of makeup do you want today?'
        return welcometext, step + 1, None, STYLE
    elif step == 1:
        for i in audio:
            if i in Style:
                text = 'Ok, let’s start.'
                STYLE = i
                return text, step + 1, None, STYLE
        if 'yes' in audio:
            text = 'Ok, let’s start.'
            return text, step + 1, None, STYLE
        elif 'no' in audio:
            text = 'OK, thank you!'
            return text, 9999, None, STYLE
        else:
            if len(Style) == 0:
                return response('no', step, STYLE)
            index = randint(0, len(Style))
            text = 'Sorry, we don’t have this kind of makeup instruction now, do you want to try ' + style_names[index] +'?'
            STYLE = style_names[index]
            return text, step, None, STYLE
    else:
        style = Style[STYLE]
        cur_step = step - 2
        index = Procesure[style["Order"][cur_step]]
        if index == "Foundation":
            if 'yes' in audio:
                foundation_type = style[index]
                foundation_tool = style["Foundation Tool"]
                text = 'Please use your '+ foundation_tool +' to apply proper amount of '+ foundation_type +' to your face uniformly.'
                return text, step+1, None, STYLE
            elif 'no' in audio:
                text = "OK, let's skip this step."
                return text, step+1, None, STYLE
            else:
                foundation_type = style[index]
                text = "Do you have any "+foundation_type +"?"
                return text, step, None, STYLE

        elif index == "Concealer":
            if 'yes' in audio:
                concealer_parts = style["Concealer Part"]
                text = 'Please use your foundation_tool to apply proper amount of '
                for part in concealer_parts:
                    text += ' '
                    text += part
                    text += ','
                text += ' to your face uniformly.'
                return text, step+1, None, STYLE
            elif 'no' in audio:
                text = "OK, let's skip this step."
                return text, step+1, None, STYLE
            else:
                text = "Do you have any Concealer with you?"
                return text, step, None, STYLE
        elif index == "Blusher":
            if color_check:
                color = style["Blusher Color"]
                text = 'Please use your brush to apply the blusher to your '
                parts = style["Blusher Part"]
                for part in parts:
                    text += ' '
                    text += part
                    text += ','
                return text, step+1, None, STYLE
            elif 'yes' in audio:
                color = style["Blusher Color"]
                text = "Ok, now let's check if you have the proper color. "
                return text, step, color
            elif 'no' in audio:
                text = "OK, let's skip this step."
                return text, step+1, None, STYLE
            else:
                text = "Do you have a Blusher?"
                return text, step, None, STYLE
        elif index == "Eyeshadow1":
            if color_check:
                text = 'Please apply the Eyeshadow to the '
                parts = style["Eyeshadow1 Part"]
                for part in parts:
                    text += part
                    text += ' and '
                text += 'of your eyes'
                return text, step+1, None, STYLE
            elif 'yes' in audio:
                color = style["Eyeshadow1 Color"]
                text = "Ok, now let's check if you have the proper color. "
                return text, step, color
            elif 'no' in audio:
                text = "OK, let's skip this step."
                return text, step+1, None, STYLE
            else:
                color = style["Eyeshadow1 Color"]
                text = "Do you have some " + color +" Eyeshadows?"
                return text, step, None, STYLE

        elif index == "Eyeshadow2":
            if color_check:
                text = 'Please apply the Eyeshadow to the '
                parts = style["Eyeshadow2 Part"]
                for part in parts:
                    text += part
                    text += ' and '
                text += 'of your eyes'
                return text, step+1, None, STYLE
            elif 'yes' in audio:
                color = style["Eyeshadow2 Color"]
                text = "Ok, now let's check if you have the proper color. "
                return text, step, color
            elif 'no' in audio:
                text = "OK, let's skip this step."
                return text, step+1, None, STYLE
            else:
                color = style["Eyeshadow2 Color"]
                text = "Do you have some " + color +" Eyeshadows?"
                return text, step, None, STYLE
        elif index == "Eyeshadow3":
            if color_check:
                text = 'Please apply the Eyeshadow to the '
                parts = style["Eyeshadow3 Part"]
                for part in parts:
                    text += part
                    text += ' and '
                text += 'of your eyes'
                return text, step+1, None, STYLE
            elif 'yes' in audio:
                color = style["Eyeshadow3 Color"]
                text = "Ok, now let's check if you have the proper color. "
                return text, step, color
            elif 'no' in audio:
                text = "OK, let's skip this step."
                return text, step+1, None, STYLE
            else:
                color = style["Eyeshadow3 Color"]
                text = "Do you have some " + color +" Eyeshadows?"
                return text, step, None, STYLE
        elif index == "Lip":
            if color_check:
                color = style["Lip Color"]
                tool = style["Lip"]
                text = 'Please apply your ' + tool + ' to your lip. '
                return text, step+1, None, STYLE
            elif 'yes' in audio:
                color = style["Lip Color"]
                text = "Ok, now let's check if you have the proper color. "
                return text, step, color
            elif 'no' in audio:
                text = "OK, let's skip this step."
                return text, step+1, None, STYLE
            else:
                tool = style["Lip"]
                text = "Do you have a "+ tool +" ?"
                return text, step, None, STYLE
        else:
            return "Error", 9999, None, STYLE

# print(response('hi', 1, "Japase"))
