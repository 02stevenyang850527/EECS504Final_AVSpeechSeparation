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

### Sample results
We put our sample results(.wav) in `samples/audio-only` and `samples/audio-visual` folder of the repo. Sample video is taken from the real-word video, which is more complicated than that in Obamanet, but both our audio-only and audio-visual models are ablet to separate the audios.

## Notes
- We put all our implementation work here, though we don't mention the details of audio-visual model because we are unable generate satisfying result because of the out limited GPU computing power and time constraint.
- modify `duration_mult` in `params.py` to a proper value. (Default: 4 for ~8 sec. videos)
