# Pytorch implementation of Show, Attend and Tell \[Chan, 2016\] and DeepSpeech models

**Note** This is an experimental code that I used for preparing a labwork. Hence the code is full of symbols "@SOL@" or so that I use for getting a template script code for the students with the [easylabwork](https://github.com/jeremyfix/easylabwork) package. By the way, if you forget about these tags, the code can be ran for training and testing speech to text models. For the students to test their implementation, they are provided with the test_implementation.py script which runs some unitary tests.

The two models I consider are the CTC based model DeepSpeech [Amodei(2015)] and the attentive Seq2Seq model Listen, Attend and Spell [Chan(2016)].

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

@inproceedings{10.5555/3045390.3045410,
author = {Amodei, Dario and Ananthanarayanan, Sundaram and Anubhai, Rishita and Bai, Jingliang and Battenberg, Eric and Case, Carl and Casper, Jared and Catanzaro, Bryan and Cheng, Qiang and Chen, Guoliang and Chen, Jie and Chen, Jingdong and Chen, Zhijie and Chrzanowski, Mike and Coates, Adam and Diamos, Greg and Ding, Ke and Du, Niandong and Elsen, Erich and Engel, Jesse and Fang, Weiwei and Fan, Linxi and Fougner, Christopher and Gao, Liang and Gong, Caixia and Hannun, Awni and Han, Tony and Johannes, Lappi Vaino and Jiang, Bing and Ju, Cai and Jun, Billy and LeGresley, Patrick and Lin, Libby and Liu, Junjie and Liu, Yang and Li, Weigao and Li, Xiangang and Ma, Dongpeng and Narang, Sharan and Ng, Andrew and Ozair, Sherjil and Peng, Yiping and Prenger, Ryan and Qian, Sheng and Quan, Zongfeng and Raiman, Jonathan and Rao, Vinay and Satheesh, Sanjeev and Seetapun, David and Sengupta, Shubho and Srinet, Kavya and Sriram, Anuroop and Tang, Haiyuan and Tang, Liliang and Wang, Chong and Wang, Jidong and Wang, Kaifu and Wang, Yi and Wang, Zhijian and Wang, Zhiqian and Wu, Shuang and Wei, Likai and Xiao, Bo and Xie, Wen and Xie, Yan and Yogatama, Dani and Yuan, Bin and Zhan, Jun and Zhu, Zhenyao},
title = {Deep Speech 2: End-to-End Speech Recognition in English and Mandarin},
year = {2016},
publisher = {JMLR.org},
abstract = {We show that an end-to-end deep learning approach can be used to recognize either English or Mandarin Chinese speech-two vastly different languages. Because it replaces entire pipelines of hand-engineered components with neural networks, end-to-end learning allows us to handle a diverse variety of speech including noisy environments, accents and different languages. Key to our approach is our application of HPC techniques, enabling experiments that previously took weeks to now run in days. This allows us to iterate more quickly to identify superior architectures and algorithms. As a result, in several cases, our system is competitive with the transcription of human workers when benchmarked on standard datasets. Finally, using a technique called Batch Dispatch with GPUs in the data center, we show that our system can be inexpensively deployed in an online setting, delivering low latency when serving users at scale.},
booktitle = {Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48},
pages = {173–182},
numpages = {10},
location = {New York, NY, USA},
series = {ICML'16}
}

```

# How to

## Get the data

The data we use are provided by the [Common voice Mozilla project](https://commonvoice.mozilla.org/en). In this project, we use the [French dataset](https://commonvoice.mozilla.org/en/datasets). You need to manually download the files and extract the archive somewhere on your drive; torchaudio cannot download the data for you since you need to accept the Mozilla terms for using the common voice datasets.


## Vocabulary

The LAS model outputs the text transcription character by character. In this implementation, dealing with the French language, the transcripts are converted to lower case and the vocabulary is [a-z, 0-9, space , period, comma, apostrophe ]. In addition, all the accents were removed (replaced with the letters without accent), the punctuation was either replaced by a period or a space, the 'œ' and 'ç' were also replaced. See the [data.CharMap](https://github.com/jeremyfix/listen-attend-and-spell/blob/main/scripts/data.py#L37) object. 


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

# Explanations

## Input data encoding: waveforms to spectrograms

The task consists in transforming a recorded waveform into the textual transcript. As input, we have the waveforms, which can be of varying duration. The neural network will not work directly on the waveform but will take as input a spectrogram. The pipeline for processing the waveform is depicted below :

![Waveform to spectrogram](https://raw.githubusercontent.com/jeremyfix/listen-attend-and-spell/main/figs/waveform_to_spectro.png)

This image has been generated with :

```python
import scripts.data as data

data.ex_waveform_spectro()
```

Computing the spectrogram with a Short Time Fourier Transform (window size of 25ms with a window step of 15 ms), we get the example spectrograms in Mel scale (with 80 filters) below :

![Spectrogram](https://raw.githubusercontent.com/jeremyfix/listen-attend-and-spell/main/figs/spectro.png)

In the code `scripts/data.py`, the data are loaded by the data.py:get_dataloaders function which :

- loads the data
- filter out the waveforms that are too short or too long in duration (by default selects only the samples between 1s. and 5s.)
- normalize the spectrograms by subtracting the mean and dividing by the variance 

A dataloader, when iterated, is returning mini batches. In the provided code, the waveforms in the minibatches are right-padded (see the BatchCollate:__call__ function) to be of the same duration. The spectrogram computation, and possibly data augmentation by frequency and time masking, is performed by the WaveformProcessor object. 

For testing the spectrogram augmentation (frequency and time masking), you can run

```python
import scripts.data as data

data.test_augmented_spectro()
```

For testing the spectrogram augmentation and padding for the minibatch construction, you can run :

```python
import scripts.data as data

data.ex_spectro()
```


The CommonVoice data are sampled at 48 kHz. Here, during the processing, the waveforms are resampled at 16 kHz. The FFT is done on windows of 25 ms and shifted by 15 ms. Hence, a sample of 5 seconds will produce an original waveform of 240.000 samples, a resampled waveform of 80.000 samples and a spectrogram of 330 samples. Given a FFT with a shift 15 ms, we also have a spectrogram at almost 67 Hz. The melscale, as in deepspeech2, we use 80 scales. The processing of the waveforms is done with [torchaudio](https://pytorch.org/audio/stable/index.html).

## Output data encoding: transcripts

The transcripts have to be encoded as integers with a specified vocabulary which is handled by the [data.CharMap](https://github.com/jeremyfix/listen-attend-and-spell/blob/05dc9aa60055b318625e40cec8141fa1fa69054c/data.py#L37) object. It adds a start of sequence tag and an end of sequence tag, converts all characters to their corresponding index in the charmap and pads the transcripts so that they are all of the same size. The padding character (so called blank character) is very specific and we use the index 0 for it which is the default of the pytorch CTC loss which is the loss used for the deepspeech model. When computed, the CTC loss needs to know where the true output sequence is ending.

So, if you take a real transcript, in French obviously, in a minibatch where the maximum transcript length is 40, like :

```
Je vais m'éclater avec des RNNs !
```

it gets encoded as :

```
[16, 27, 22, 1, 39, 18, 26, 36, 1, 30, 3, 22, 20, 29, 18, 37, 22, 35, 1, 18, 39, 22, 20, 1, 21, 22, 36, 1, 35, 31, 31, 36, 1, 5, 2, 0, 0, 0, 0, 0]
```

If you decode it back into characters, you then get a slightly different sentence :

```
¶je vais m'eclater avec des rnns .¦¬¬¬¬¬
```

For the conversion, I decided to convert to lower case, to remove the accents, to convert the punctuation to either a period or a space. The charmap contains 44 symbols.

To run this test, run the 

```python

import scripts.data as data

data.test_charmap()

```

Note: At the time of writing, there is some issues with the encoding of some special characters (like œ or ç)are not correctly handled.

## CTC Model 

The CTC model we implement is made of a convolutional network followed by a recurrent network then followed by a linear layer producing a probability distribution over the vocabulary which contains the special blank symbol introduced by A. Graves in his work on [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf). To see more about the architecture, check the [models.py](https://github.com/jeremyfix/listen-attend-and-spell/blob/main/scripts/models.py#L52). The implementation of this repository considers bidirectional recurrent layers, which makes sense if you have access to the full recording before trying to transcribe it.

The loss is then the CTC loss which computes the probability of a target sequence by marginalizing over all the possible sequences that contain the blank, empty symbol. The CTC loss is an option for situation where the input sequence is always longer than the input sequence. If the input sequence is of length 5 and the target sequence is, say, 'a','b','c', then the probability of this target sequence is computed by summing the probabilities that your model assign to all the following sequences :

- <blank>,<blank>,a,b,c
- <blank>, a, <blank>, b, c
- <blank>, a, b, <blank>, c
- <blank>, a, b, c, <blank>
- a, <blank>,<blank>,b, c
- a, <blank>,b, <blank>, c
- ...

There is a very interesting article on [distill.pub](https://distill.pub/2017/ctc/) on the CTC loss.

**TODO** : example with the pytorch CTC loss

Decoding from this model is specific as well and performed with beam search by considering the most likely sequence hypothesis.


## Seq2Seq with attention

**TO BE DONE :-) **

# References

You might be interested in :

- [Openseq2Seq from nvidia](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html)
- [Sean Naren deepspeech](https://github.com/SeanNaren/deepspeech.pytorch)
- [Coqui STT](https://stt.readthedocs.io/en/latest/)
- [Mozilla deepspeech](https://github.com/mozilla/DeepSpeech)
