# Pytorch implementation of Show, Attend and Tell \[Chan, 2016\]

This is a tentative pytorch reimplementation of the image captioning network Listen, Attend and Spell.


```
@INPROCEEDINGS{7472621,
  author={W. {Chan} and N. {Jaitly} and Q. {Le} and O. {Vinyals}},
  booktitle={2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Listen, attend and spell: A neural network for large vocabulary conversational speech recognition}, 

  year={2016},
  volume={},
  number={},
  pages={4960-4964},
  doi={10.1109/ICASSP.2016.7472621}
}
```

# How to

## Get the data

The data we use are provided by the [Common voice Mozilla project](https://commonvoice.mozilla.org/en). In this project, we use the [French dataset](https://commonvoice.mozilla.org/en/datasets). You need to manually download the files and extract the archive somewhere on your drive; torchaudio cannot download the data for you since you need to accept the Mozilla terms for using the common voice datasets.

Computing the spectrogram with a Short Time Fourier Transform (window size of 25ms with a window step of 15 ms), we get the example spectrograms in Mel scale below :

![Spectrogram](https://github.com/jeremyfix/listen-attend-spell/blob/main/figs/spectro.png?raw=true)

## Testing the notebooks

```
pipenv install jupyterlab matplotlib torchaudio
pipenv run jupyter lab
```
