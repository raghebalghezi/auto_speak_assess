from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import math
import librosa
import numpy
import json
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def extract_words(res):
    jres = json.loads(res)
    if not 'result' in jres:
        return []
    words = jres['result']
    return words

def transcribe_words(recognizer, bytes):
    results = []
    chunk_size = 4000
    for chunk_no in range(math.ceil(len(bytes)/chunk_size)):
        start = chunk_no*chunk_size
        end = min(len(bytes), (chunk_no+1)*chunk_size)
        data = bytes[start:end]

        if recognizer.AcceptWaveform(data):
            words = extract_words(recognizer.Result())
            results += words
    results += extract_words(recognizer.FinalResult())
    transcription =  ' '.join(map(str, [item["word"] for item in results]))
    return transcription

def asr_wav2vec(audio, sr, model, processor):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    #model = model.to(device)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]


def asr_kaldi(kaldi_path, audio, sr):
    SetLogLevel(-1)
    if not os.path.exists(kaldi_path):
        print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
        exit (1)    
    model = Model(kaldi_path)
    rec = KaldiRecognizer(model, sr)#wf.getframerate())#, '["oh one everyone three four five six recording eight nine zero", "[unk]"]')
    return transcribe_words(rec, numpy.int16(audio * 32768).tobytes())

def asr(kaldi_path, w2v_model, w2v_processor, audio=None, sr=None):
    if audio is None and sr is None:
        print('Error! No audio loaded!')
        #audio, sr = get_audio() #a function for recording audio
    elif type(audio) is str:
        audio, sr = librosa.load(audio, sr=16000)
    if sr != 16000:
        #audio = librosa.resample(np.array(audio/32767.0, dtype=np.float32), sr, 16000)
        audio = librosa.resample(audio, sr, 16000)
        sr = 16000
    transcript_kaldi=asr_kaldi(kaldi_path=kaldi_path, audio=audio, sr=sr)
    transcript_wav2vec=asr_wav2vec(audio=audio, sr=sr, model=w2v_model, processor=w2v_processor)
    return transcript_kaldi, transcript_wav2vec