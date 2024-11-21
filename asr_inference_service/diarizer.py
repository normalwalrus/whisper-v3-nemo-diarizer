from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.utils import nemo_logging
from typing import List, Union

import logging
import torch
import pandas as pd

logger_nemo = logging.getLogger('nemo_logger')
logger_nemo.disabled = True

class DiarInference:
    '''
    Diar inference class
    '''
    def __init__(self, model_path: str, device: Union[str, List[int]], accelerator: str):
        
        self.map_location = torch.device(f'cuda:{device[0]}' if accelerator == 'gpu' else 'cpu')
        self.diar_model = NeuralDiarizer.from_pretrained(model_path).to(self.map_location)

    def diarize(self, audio_path: str) -> pd.DataFrame:
        annotation = self.diar_model(audio_path, num_workers=0, batch_size=16)
        rttm=annotation.to_rttm()
        df = pd.DataFrame(columns=['start_time', 'end_time', 'speaker', 'text'])
        lines = rttm.splitlines()
        if len(lines) == 0:
            df.loc[0] = 0, 0, 'No speaker found'
            return df
        start_time, duration, prev_speaker = float(lines[0].split()[3]), float(lines[0].split()[4]), lines[0].split()[7]
        end_time = float(start_time) + float(duration)
        df.loc[0] = start_time, end_time, prev_speaker, ''

        for line in lines[1:]:
            split = line.split()
            start_time, duration, cur_speaker = float(split[3]), float(split[4]), split[7]
            end_time = float(start_time) + float(duration)
            if cur_speaker == prev_speaker:
                df.loc[df.index[-1], 'end_time'] = end_time
            else:
                df.loc[len(df)] = start_time, end_time, cur_speaker, ''
            prev_speaker = cur_speaker

        return df
    

if __name__ == '__main__':
    
    if torch.cuda.is_available():
        DEVICE = [0]  # use 0th CUDA device
        ACCELERATOR = 'gpu'
    else:
        DEVICE = 1
        ACCELERATOR = 'cpu'
    
    model_name = 'diar_msdd_telephonic'
    path_to_example = 'example/steroids_120sec.wav'
    
    diar_model = DiarInference(model_name, DEVICE, ACCELERATOR)
    df = diar_model.diarize(path_to_example)
    
    print(df)