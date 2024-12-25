# Triton Inference Server + Silero VAD Client Setup

## 1. Run Triton Inference Server Locally

Start the Triton server using Docker with the ONNX Silero VAD model:

```bash
docker run --rm \ -gpus all
  -v /path/to/model_repository:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:23.11-py3 \
  tritonserver --model-repository=/models
```
## 2. Client set up
Once the triton server is running , the client_script.py can be used to interact with the Silero-VAD model. The required packages can be installed using
```bash

pip install requirements.txt
```
```bash

python client.py --audio_file path/to/audio.wav
```
The client script can take in one audio.wav as input.
If no audio file , the model will generate output for a random input of shape (1,512). The output will be a list of dictionaries containing the start and end time stamps where voice activity is detected
