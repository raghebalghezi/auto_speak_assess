from flask import Flask, render_template, request, redirect
import speech_recognition as sr
from  fluency import calc_fluency
from werkzeug.utils import secure_filename
import glob, os
import random


for f in glob.glob("upload/*"):
    os.remove(f)

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    fluency_info = ""
    task_achievement = ""
    pronunciation = ''
    range = ""
    accuracy = ""
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
        
        r = sr.Recognizer()
        with sr.AudioFile(f'upload/{filename}') as source:
            audio = r.record(source)  # read the entire audio file

        transcript = r.recognize_google(audio, show_all=False, language="fi")

        fluency_info = calc_fluency()
        task_achievement = f" {random.choice([1,2,3])} out of 3 (unreal score)"
        range = f"{random.choice([1,2,3])} out of 3 (unreal score)"
        accuracy = f"{random.choice([1,2,3])} out of 3 (unreal score)"
        pronunciation = f"{random.choice([1,2,3])} out of 3 (unreal score)"


    return render_template('index.html', transcript=transcript, \
        fluency=fluency_info, taskAchievement=task_achievement, range=range, \
            accuracy=accuracy, pronunciation=pronunciation)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)