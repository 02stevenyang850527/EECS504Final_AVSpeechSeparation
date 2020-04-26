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

## model
We provide a script `download_data_model.sh` to get our training model and sample data.

#### Usage
After downloading the model: 
- Testing on audio-only model: `python3 vid_sep.py -t -m unet -s 1000 -r 1`
- Testing on audio-visual model: `python3 vid_sep.py -t -m vidsep -s 1000 -r 1`

### Demo
![Sample Audio](samples/orig.wav)
## Notes
Currently, we only provide the audio-only model (U-Net) though the code is ready to run both audio-only and audio-visual model.
