

import tritonclient.http as httpclient
import numpy as np
import wave
import argparse
import librosa
import sys
import os
sys.stdout = open(os.devnull, 'w')

# Now any print statements will not show up


# Load the audio file
# Triton server URL
# Replace localhost with your WSL2 IP address
url = "172.18.250.117:8000" 
model_name = "silero_vad"  # Replace with the actual model name
model_version="1"
def read_audio(audio_file=None):
    """Reads the audio file and returns audio data as a numpy array."""
    if audio_file is None:
        audio=np.random.randn(1, 512).astype(np.float32)
    else:
        audio, sr = librosa.load(audio_file, sr=16000)
        audio=audio.reshape(1,audio.shape[0]).astype(np.float32)
    chunk_size = 512
    chunks = [audio[:,i:i + chunk_size] for i in range(0, audio.shape[1], chunk_size)]
    # If the last chunk is smaller than 512 samples, pad it
    if len(chunks[-1]) < chunk_size:
        temp_chunk=np.zeros((chunks[0].shape[0],512))
        temp_chunk[:,:chunks[-1].shape[1]]=chunks[-1]
        chunks[-1] = temp_chunk        
    return chunks


def run_inference(chunks):
    """Runs inference on the Triton server with the provided audio data."""
    # Create a Triton HTTP client
    client = httpclient.InferenceServerClient(url=url, verbose=True)
    chunk_duration = 512 / 16000

    # Prepare input tensor (modify the shape/dimensions as per the Silero VAD model config)
    
    # Request output
    output_name = "output"
    output_data = httpclient.InferRequestedOutput(output_name)
    stateN_output = httpclient.InferRequestedOutput("stateN") # Adjust output name
    state_data = np.zeros((2,1,128)).astype(np.float32)
    # Prepare sr input (shape: scalar or 1D)
    sr_data = np.array([16000]).astype(np.int64)
    inputs = [
    httpclient.InferInput("input", (1,512), "FP32"),
    httpclient.InferInput("state", state_data.shape, "FP32"),
    httpclient.InferInput("sr", sr_data.shape, "INT64")
    ]
    output_probabilities=[]
    time_stamps=[]
    # Send the request to Triton and get the result
    f=0
    for i, chunk in enumerate(chunks[:-1]):
        # Calculate the starting time of the chunk in the original audio
        #print(chunk.shape)
        inputs[0].set_data_from_numpy(chunk)
        inputs[1].set_data_from_numpy(state_data)
        inputs[2].set_data_from_numpy(sr_data)

        start_time = i * chunk_duration
        result = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=[output_data, stateN_output],
        model_version=model_version
        )
        output_result = result.as_numpy(output_name)
        stateN_result = result.as_numpy("stateN")
        #print(list(output_result)[0])
        
        output_probabilities.append(list(output_result)[0].item())
       
        if list(output_result)[0].item()>0.5:
            if f==1:
                time_stamps[-1]["end"]=start_time+chunk_duration
            else:
                time_stamps.append({"start":start_time,"end":start_time+chunk_duration})
            #time_stamps.append(start_time)
            f=1
        else:
            f=0
        state_data=stateN_result
    return time_stamps


def main(audio_file=None):
    # Load and preprocess the audio file
    chunks = read_audio(audio_file)
    
    # Run inference
    vad_output = run_inference(chunks)

    # Output VAD result
    sys.stdout = sys.__stdout__
    print(f"VAD Output: {vad_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client script for Silero VAD model")
    parser.add_argument(
        "--audio_file",
        type=str,  # Replace with a default audio file
        help="Path to the input audio file (.wav format)"
    )
    args = parser.parse_args()

    # Run the client with the provided or default audio file
    main(args.audio_file)
