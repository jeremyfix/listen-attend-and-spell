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

Computing the spectrogram with a Short Time Fourier Transform (window size of 25ms with a window step of 15 ms), we get the example spectrograms in Mel scale (with 80 filters) below :

![Spectrogram](https://raw.githubusercontent.com/jeremyfix/listen-attend-and-spell/main/figs/spectro.png)

The pipeline for processing the waveform is depicted below :

![Waveform to spectrogram](https://raw.githubusercontent.com/jeremyfix/listen-attend-and-spell/main/figs/waveform_to_spectro.png)

## Vocabulary

The LAS model outputs the text transcription character by character. In this implementation, dealing with the French language, the transcripts are converted to lower case and the vocabulary is [a-z, 0-9, space , period, comma, apostrophe ]. In addition, all the accents were removed (replaced with the letters without accent), the punctuation was either replaced by a period or a space, the 'œ' and 'ç' were also replaced. See the [data.CharMap](https://github.com/jeremyfix/listen-attend-and-spell/blob/05dc9aa60055b318625e40cec8141fa1fa69054c/data.py#L37) object. 


## Training

For training with the defaults :

```
python3 main.py train
```

To see options that can be customized : 

```
python3 main.py train --help
```

For debugging purpose, e.g. ensuring the training pipeline works, you can experiment on a small subset of the datasets, using a small model

```
python3 main.py train --nhidden_spell 8 --nhidden_listen 8 --dim_embed 24 --debug
```

You can consider invoking `python3 -W ignore ...` since at the time of writing, we otherwise get several UserWarning on deprecated torch.rfft which make the terminal output unreadable.

