#!/usr/bin/env python
# -*- coding: utf-8 -*-

import speech_recognition as sr

def listner():
    mic = sr.Recognizer()
    with sr.Microphone() as source:
        mic.adjust_for_ambient_noise(source)
        print('Speech to me..')
        audio = mic.listen(source)


    try:

        f = mic.recognize_google(audio, language='pt-BR')
        print('Did you mean '+ f)
    except sr.UnknownValueError:
        print('I dont understanding')
    return f
f = listner()
        