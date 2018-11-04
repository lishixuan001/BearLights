import speech_recognition as sr
from gtts import gTTS
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

Japanese = {"Style": "Japanese", "Steps": 8, "Order": None, "Foundation":"Liquid Foundation", "Foundation Tool":"Powder Puff",
    "Concealer": True, "Concealer Part": ["Eyes"], "Blusher": True, "Blusher Color": "Pink", "Blusher Part": ["Risorius"],
    "Eyeshadow1": "Earch Tone", "Eyeshadow1 Part":["Upper Eyelid"], "Eyeshadow2": "Earch Tone",
    "Eyeshadow2 Part":["Base", "Outer Corner"], "Eyeshadow3": "Purple",
    "Eyeshadow3 Part":["Base", "Outer Corner"], "Eyeliner": "Pencil",
    "Eyeliner Note": "Going up a little bit towards the brow to produce lifted effect.",
        "Fake Lashes": True, "Mascara": True, "Lip": "Lipstick", "Lip Color": "Pink"}
Style = {"Japanese": Japanese}


#Initialize the audio files
welcometext = 'Hi, what kind of makeup do you want today?'
language = 'en'
welcome = gTTS(text=welcometext, lang=language, slow=False)
welcome.save("welcome.mp3")

done = "Okay, I'll see what I can do for you."
done = gTTS(text=done, lang=language, slow=False)
done.save("request_received.mp3")

request_unclear = "I'm not sure what you said."
request_unclear = gTTS(text=request_unclear, lang=language, slow=False)
request_unclear.save("request_unclear.mp3")

#initialize the stopwords
stop_words = set(stopwords.words('english'))

#initialize the stemmer
ps = PorterStemmer()

r = sr.Recognizer()
with sr.Microphone() as source:
    os.system("afplay welcome.mp3")
    audio = r.listen(source)
    try:
        text = [ps.stem(i) for i in r.recognize_google(audio).split(" ") if not i in stop_words]
        if text[0] in Style:
            style = text[0]
            temp = 'Ok, let’s start.'
            temp = gTTS(text=temp, lang=language, slow=False)
            temp.save("temp.mp3")
            os.system("afplay temp.mp3")
        else:
            index = random(range(len(Style)))
            temp = 'Sorry, we don’t have a Half_blood makeup instruction now, do you want to try' + Style[index] +'?'
            temp = gTTS(text=temp, lang=language, slow=False)
            temp.save("temp.mp3")
            os.system("afplay temp.mp3")
            audio = r.listen(source)
            style = Style[index]
            text = [ps.stem(i) for i in r.recognize_google(audio).split(" ") if not i in stop_words]
            if "yes" not in text:
                return
    except sr.UnknownValueError:
        os.system("afplay request_unclear.mp3")
    instruction = Style[style]
    for i in range(instruction[Steps]):
        try:
            audio = r.listen(source)
            text = [ps.stem(i) for i in r.recognize_google(audio).split(" ") if not i in stop_words]
            os.system("afplay request_received.mp3")
        except sr.UnknownValueError:
            os.system("afplay request_unclear.mp3")
