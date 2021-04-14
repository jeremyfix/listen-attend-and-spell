# Pytorch implementation of Show, Attend and Tell \[Chan, 2016\] and DeepSpeech models

**Note** This is an experimental code that I used for preparing a labwork. Hence the code is full of symbols "@SOL@" or so that I use for getting a template script code for the students with the [easylabwork](https://github.com/jeremyfix/easylabwork) package. By the way, if you forget about these tags, the code can be ran for training and testing speech to text models.

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

Computing the spectrogram with a Short Time Fourier Transform (window size of 25ms with a window step of 15 ms), we get the example spectrograms in Mel scale (with 80 filters) below :

![Spectrogram](https://raw.githubusercontent.com/jeremyfix/listen-attend-and-spell/main/figs/spectro.png)

The pipeline for processing the waveform is depicted below :

![Waveform to spectrogram](https://raw.githubusercontent.com/jeremyfix/listen-attend-and-spell/main/figs/waveform_to_spectro.png)

This transformation of waveforms to logmel spectrograms is done in the WaveProcessor object.

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

# References

You might be interested in :

- [Openseq2Seq from nvidia](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html)
- [Sean Naren deepspeech](https://github.com/SeanNaren/deepspeech.pytorch)
