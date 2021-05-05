import parselmouth
from parselmouth.praat import call, run_file
# import numpy as np


# sound = "freeSpeech.w

def calc_fluency():
    first_script = r"SyllableNucleiv3.praat"
    path= r"./upload/*.wav"

    objects_1st = run_file(first_script, path, "None", -25, 2,0.3, True, "English", 1.0, "Praat Info window", "OverWriteData", True, capture_output=True)

    objects, info = objects_1st

    header, file_info = info.strip().split('\n')[:2]
    info_dict = {}
    for i in zip(header.split(','), file_info.split(',')):
        info_dict[i[0].strip()] = i[1].strip()
    return info_dict