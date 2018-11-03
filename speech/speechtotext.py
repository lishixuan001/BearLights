import speech_recognition as sr
import nltk
from gtts import gTTS
import os

welcometext = 'Welcome to bear light! What would you like to do?'
language = 'en'
welcome = gTTS(text=welcometext, lang=language, slow=False)
welcome.save("welcome.mp3")

done = "Okay, I'll see what I can do for you."
done = gTTS(text=done, lang=language, slow=False)
done.save("request_received.mp3")

request_unclear = "I'm not sure what you said."
request_unclear = gTTS(text=request_unclear, lang=language, slow=False)
request_unclear.save("request_unclear.mp3")



r = sr.Recognizer()
with sr.Microphone() as source:
    os.system("afplay welcome.mp3")
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        os.system("afplay request_received.mp3")
        print (text)
    except sr.UnknownValueError:
        os.system("afplay request_unclear.mp3")
