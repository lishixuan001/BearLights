import speech_recognition as sr
from gtts import gTTS
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer


#Initialize the audio files
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
        print(text)
        os.system("afplay request_received.mp3")
    except sr.UnknownValueError:
        os.system("afplay request_unclear.mp3")
