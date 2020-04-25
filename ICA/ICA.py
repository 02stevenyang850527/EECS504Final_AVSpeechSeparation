import numpy as np
import sys
import scipy.io.wavfile as wavfile
#########################
### Utility Functions ###
#########################
def mix_audio(wav_list, sr=16000, output_name='mixed.wav'):
    audio_num = len(wav_list)
    source = np.zeros((sr*3, audio_num))
    for idx, file_name in enumerate(wav_list):
        s, wav = wavfile.read(file_name)
        source[:, idx] = wav[:sr*3]
    sample = np.random.random((audio_num, audio_num))
    mixed = source @ sample
    wavfile.write(output_name, sr, mixed[:, 0])
    return source, mixed

def normalize(data):
    return 0.99 * data / np.max(np.abs(data))

def sigmoid(x):
    sig0 = 1 / (1 + np.exp(-x))
    sig1 = np.exp(x) / (1 + np.exp(x))
    return np.where(x >= 0, sig0, sig1)
###########################
###         ICA         ###
###########################
def train_ICA(X):
    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    lr = 1e-3
    max_iter = 100
    M, N = X.shape
    W = np.eye(N)
    for it in range(1, max_iter+1):
        for x_i in X:
            tmp = 1 - 2 * sigmoid(np.dot(W, x_i.T))
            W += lr * (np.outer(tmp, x_i) + np.linalg.inv(W.T))
        if it % 10 == 0:
            print("Iteration: %d"%it)
    """
    Sanity Check
    """
    print("W:")
    print(W)
    return W

def test_ICA(X, W):
    return  np.dot(X, W.T)

def main():
    sdrs, sirs, sars, pesqs = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
    for idx in range(1, 11):
        sr = 16000
        wav_list = ['data/3.wav', sys.argv[1]+str(idx)+'.wav']
        src, mixed = mix_audio(wav_list)
        """ self-defined ICA
        ## train
        W = train_ICA(mixed) 
        ## test
        S = normalize(test_ICA(mixed, W))
        """
        from sklearn.decomposition import FastICA
        W = FastICA(n_components=len(wav_list))
        S = W.fit_transform(mixed)
        ## write audio into file
        for i in range(len(wav_list)):
            output_name = "result/%d.wav"%(i+1)
            wavfile.write(output_name, sr, S[:, i])
        """
        Caluculate sdr, sir, sar
        """
        from mir_eval.separation import bss_eval_sources
        sdr, sir, sar, _ = bss_eval_sources(S.T, src.T)         ## shape = (channels, samples)
        
        print("SDR: ", sdr)  ## np.array(channels, )
        print("SIR: ", sir)
        print("SAR: ",sar)
        from pypesq import pesq
        pesq_score = pesq(src[:, 0], S[:, 0], fs=16000)
        print("PESQ: ", pesq_score)
        sdrs[idx-1] = sdr[0]
        sirs[idx-1] = sir[0]
        sars[idx-1] = sar[0]
        pesqs[idx-1] = pesq_score
    print("SDR: ", np.mean(sdrs))  ## np.array(channels, )
    print("SIR: ", np.mean(sirs))
    print("SAR: ", np.mean(sars))
    print("PESQ: ", np.mean(pesqs))

def sample_main():
    sr = 16000
    wav_list = ['data/1.wav', 'data/2.wav', 'data/3.wav', 'data/4.wav', 'data/5.wav']
    src, mixed = mix_audio(wav_list)
    from sklearn.decomposition import FastICA
    W = FastICA(n_components=len(wav_list))
    S = W.fit_transform(mixed)
    ## write audio into file
    for i in range(len(wav_list)):
        output_name = "result/%d.wav"%(i+1)
        wavfile.write(output_name, sr, S[:, i])
    """
    Caluculate sdr, sir, sar
    """
    from mir_eval.separation import bss_eval_sources
    sdr, sir, sar, _ = bss_eval_sources(S.T, src.T)         ## shape = (channels, samples)
    print("SDR: ", sdr)  ## np.array(channels, )
    print("SIR: ", sir)
    print("SAR: ",sar)
    from pypesq import pesq
    pesq_score = pesq(src, S, fs=16000)
    print("PESQ: ", pesq_score)

if __name__ == '__main__':
    ## for displaying
    sample_main()
    ## for running experiment
    # main()