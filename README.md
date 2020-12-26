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

![Spectrogram](https://raw.githubusercontent.com/jeremyfix/listen-attend-and-spell/main/figs/specro.png)

## Vocabulary

The LAS model outputs the text transcription character by character. In this implementation, dealing with the French language, the transcripts are converted to lower case and the vocabulary is [a-z, 0-9, <space> , <period>, <comma>, <apostrophe>]. In addition, all the accents were removed (replaced with the letters without accent), the punctuation was either replaced by a period or a space, the 'œ' and 'ç' were also replaced. See the [data.CharMap](https://github.com/jeremyfix/listen-attend-and-spell/blob/05dc9aa60055b318625e40cec8141fa1fa69054c/data.py#L37) object. 

## Testing the notebooks

```
pipenv install jupyterlab matplotlib torchaudio
pipenv run jupyter lab
```
