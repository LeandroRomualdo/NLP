#!/usr/bin/env python
# -*- coding: utf-8 -*-

import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

def get_audio(audio):
	tts = gTTS(audio,lang='pt-br')
	tts.save('hello.mp3')
	print('I m listen to you speech...')
	playsound('speech.mp3')

def list_mic():
    microfone = sr.Recognizer()
    
    with sr.Microphone() as source:
        microfone.adjust_for_ambient_noise(source)
        print('Speech to me...')
        audio = microfone.listen(source)
    try:
        f = microfone.recognize_google(audio,language='pt-BR')
        print('Did you mean ' + f)

    except sr.UnkownValueError:
        print('I dont understanding..')

    return f

f = get_audio()
list_mic(f)