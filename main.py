from flask import Flask, render_template, request, redirect
from  fluency import calc_fluency
from werkzeug.utils import secure_filename
import glob, os
import random
from utils import *


# for f in glob.glob("upload/*"):
#     os.remove(f)

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript_kaldi = ""
    transcript_wav2vec = ""
    fluency_info = ""
    task_achievement = ""
    pronunciation = ''
    range = ""
    accuracy = ""

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    w2v_model = Wav2Vec2ForCTC.from_pretrained('wav2vec2-large-100K-digitala-freeform').to(device)
    w2v_processor = Wav2Vec2Processor.from_pretrained("wav2vec2-large-100K-digitala-freeform")
    kaldi_path='digitala-freeform-kaldi'

    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        filename = secure_filename(file.filename)
        # print(file, type(file))
        file.save(f'upload/{filename}')

        if file.filename == "":
            return redirect(request.url)
        
        # r = sr.Recognizer()
        # with sr.AudioFile(f'upload/{filename}') as source:
        #     audio = r.record(source)  # read the entire audio file

        # transcript = r.recognize_google(audio, show_all=False, language="fi")
        audio, sr = librosa.load(os.path.join('upload',filename))
        transcript_kaldi, transcript_wav2vec = asr(kaldi_path=kaldi_path, w2v_model=w2v_model, \
            w2v_processor=w2v_processor, audio=audio, sr=sr)

        fluency_info = calc_fluency()
        task_achievement = f" {random.choice([1,2,3])} out of 3 (unreal score)"
        range = f"{random.choice([1,2,3])} out of 3 (unreal score)"
        accuracy = f"{random.choice([1,2,3])} out of 3 (unreal score)"
        pronunciation = f"{random.choice([1,2,3])} out of 3 (unreal score)"

        print("KALDI",transcript_kaldi)
        print("WAV2VEC2",transcript_wav2vec)
    return render_template('index.html', transcript_kaldi=transcript_kaldi, \
        transcript_wav2vec=transcript_wav2vec, \
        fluency=fluency_info, taskAchievement=task_achievement, range=range, \
            accuracy=accuracy, pronunciation=pronunciation)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)