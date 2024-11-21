"""
API module for ASR service.

This module provides the FastAPI application for performing ASR.
"""

import io
import json
import logging
import os
import tempfile
import shutil
from typing import List

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.status import HTTP_200_OK

from asr_inference_service.model import ASRModelForInference
from asr_inference_service.schemas import ASRResponse, HealthResponse

SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8080

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

logging.getLogger('nemo_logger').setLevel(logging.ERROR)

app = FastAPI()
model = ASRModelForInference(
    model_dir=os.environ["PRETRAINED_MODEL_DIR"],
    sample_rate=int(os.environ["SAMPLE_RATE"]),
    device=os.environ["DEVICE"]
)

class AudioData(BaseModel):
    array: list


@app.get("/", status_code=HTTP_200_OK)
async def read_root():
    """Root Call"""
    return {"message": "This is an ASR service."}


@app.get("/health")
async def read_health() -> HealthResponse:
    """
    Check if the API endpoint is available.

    This endpoint is used by Docker to check the health of the container.
    """
    return {"status": "HEALTHY"}


@app.post("/v1/transcribe", response_model=ASRResponse)
async def transcribe(data: Request):
    """Function call to takes in an audio file as bytes, and executes model inference"""
    data = await data.json()

    transcription = model.infer(data["array"], 16000)

    return {"transcription": str(transcription)}


@app.post("/v1/transcribe_filepath", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")

    # Receive the audio bytes from the request
    audio_bytes = file.file.read()

    # load with soundfile, data will be a numpy array
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    transcription = model.infer(data, samplerate)

    return {"transcription": str(transcription)}

@app.post("/v1/diarize_filepath", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, saves it as a temp .wav file and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")
    
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        # Write the content of the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        
        temp_file_path = temp_file.name
        transcription = model.diar_inference(temp_file_path)

    return {"transcription": str(transcription)}


@app.post("/v1/transcribe_vad", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")

    # Receive the audio bytes from the request
    audio_bytes = file.file.read()

    # load with soundfile, data will be a numpy array
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    transcription = model.vad_inference(data, samplerate)

    return {"transcription": str(transcription)}


@app.post("/v1/transcribe_diarize", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")

    # Receive the audio bytes from the request
    audio_bytes = file.file.read()

    # load with soundfile, data will be a numpy array
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    transcription = model.diar_inference(data, samplerate)

    return {"transcription": str(transcription)}


def start():
    """Launched with `start` at root level"""
    uvicorn.run(
        "asr_inference_service.main:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=False,
    )


if __name__ == "__main__":
    start()
