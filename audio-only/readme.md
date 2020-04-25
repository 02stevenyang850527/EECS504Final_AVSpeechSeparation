This repository is the Pytorch version of reproduction for the [paper](https://arxiv.org/pdf/1804.03641.pdf).
Here is the original [project page](http://andrewowens.com/multisensory)

## Requirements
A linux machine with Nvidia GPU and CUDA, python3 ebvironment with following packages:  
- pillow
- librosa
- Pytorch
- torchaudio
- torchvision
- numpy
- scipy
- tqdm

## Audio-only model
We provide a script `download_data_model.sh` to get our training model and sample data.

## Usage
After download the model: 
- Testing: python3 vid_sep.py -t -m unet -s 1000 -r 1

## Notes
Currently, we only provide the audio-only model (U-Net) though the code is ready to run both audio-only and audio-visual model.
