import os
import librosa
import math
import json

SAMPLE_RATE = 44100
SAMPLES_FILE = SAMPLE_RATE * 30 
def create_mfccs(data_path,json_path,mffc_n=13,fft_n=4096,hop_len=1024,segment_n=10):
    
    data = {
        "map":[],
        "mfcc":[],
        "label":[]
    }

    samples_segment = SAMPLES_FILE / segment_n
    mfccs_segment = math.ceil(samples_segment / hop_len)

    for i,(path,dir,file) in enumerate(os.walk(data_path)):

        if path != data_path:
            label = path.split('/')[-1]
            data["map"].append(label)
            print("\n Progress: {}".format(label))

            for f in file:
                f_path = os.path.join(path,f)
                signal,sample_rate = librosa.load(f_path,sr=SAMPLE_RATE)

                for j in range(segment_n):
                    
                    start = samples_segment * j
                    end = start + samples_segment

                    mfcc = librosa.feature.mfcc(signal[int(start):int(end)],sample_rate,n_fft=fft_n,n_mfcc=mffc_n,hop_length=hop_len)
                    mfcc = mfcc.T
                    
                    if len(mfcc) == mfccs_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["label"].append(i-1)

    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)                    

if __name__ == "__main__":
    create_mfccs("genres","data.json")

