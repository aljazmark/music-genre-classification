from tensorflow import keras
import librosa
import numpy as np
import math
import os
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
MAPPING = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

SAMPLE_RATE = 44100
SAMPLE_LENGTH = 3.0

def process_track(path):
    track=[]
    signal,sample_rate = librosa.load(path,sr=SAMPLE_RATE)
    cut = (len(signal)/SAMPLE_RATE)%SAMPLE_LENGTH
    end = round(len(signal)-SAMPLE_RATE*cut)
    signal=signal[0:end]
    segment_n = math.floor(librosa.get_duration(signal,sample_rate)/SAMPLE_LENGTH)
    samples_file= sample_rate * round(librosa.get_duration(signal,sample_rate)) 
    samples_segment = samples_file / segment_n
    mfccs_segment = math.ceil(samples_segment / 1024)
    for j in range(segment_n):
                    
                    start = samples_segment * j
                    end = start + samples_segment

                    mfcc = librosa.feature.mfcc(signal[int(start):int(end)],sample_rate,n_fft=4096,n_mfcc=13,hop_length=1024)
                    mfcc = mfcc.T
                    mfcc = mfcc[...,np.newaxis]   
                    if len(mfcc) == mfccs_segment:
                        track.append(mfcc)
    return track
def predict(model,track):
    segs = len(track)
    sums = np.zeros(10)
    for tr in track:
        tr = tr[np.newaxis, ...]
        predictions = model.predict(tr)
        predictions = predictions[0]
        for i in range(10):
            sums[i] = sums[i] + predictions[i]          
    for i in range(10):
            sums[i] = sums[i]/segs        
    return sums

def draw_figure(can,fig):
    fig_can = FigureCanvasTkAgg(fig,can)
    fig_can.draw()
    fig_can.get_tk_widget().pack(side="top", fill="both", expand=1)
    return fig_can
def clear_figure(fig):
    fig.get_tk_widget().forget()
def draw_plot(data):
    inds = np.argpartition(data, -3)[-3:]
    genres = []
    vals = []
    for x in range(3):
        genres.append(MAPPING[inds[x]])
        vals.append(data[inds[x]])  
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    ax = fig.subplots(1,1)
    y_pos = np.arange(3)
    ax.barh(y_pos, vals, align='center')
    ax.set_yticks(y_pos, labels=genres)
    ax.invert_yaxis() 
    ax.set_xlabel('Probability')
    return fig
if __name__ == "__main__":
    matplotlib.use("TkAgg")
    model = keras.models.load_model('model')
    figg = None
    files_list = [
        [
            sg.Text("Select a folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
    ]
    prediction = [
        [sg.Text("Choose a track")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Canvas( key="-CANVAS-")],
    ]

    layout = [
        [
            sg.Column(files_list),
            sg.VSeperator(),
            sg.Column(prediction),
        ]
    ]

    window = sg.Window("Music Genre Classification", layout, finalize=True)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower()
            ]
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":
            try: 
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                track = process_track(filename)
                guess = predict(model,track)

                window["-TOUT-"].update("Classification: "+MAPPING[np.argmax(guess)])
                fig = draw_plot(guess)
                if figg:
                    clear_figure(figg)
                figg = draw_figure(window["-CANVAS-"].TKCanvas, fig)
            except:
                window["-TOUT-"].update("Problem processing track")
                if figg:
                    clear_figure(figg)

    window.close()
