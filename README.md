# Pytorch implementation of Show, Attend and Tell \[Xu, 2015\]

This is a tentative pytorch reimplementation of the image captioning network Show, Attend and Tell 


```
@InProceedings{pmlr-v37-xuc15,
	title = {Show, Attend and Tell: Neural Image Caption Generation with Visual Attention}, author = {Kelvin Xu and Jimmy Ba and Ryan Kiros and Kyunghyun Cho and Aaron Courville and Ruslan Salakhudinov and Rich Zemel and Yoshua Bengio},
	booktitle = {Proceedings of the 32nd International Conference on Machine Learning},
	pages = {2048--2057},
	year = {2015},
	editor = {Francis Bach and David Blei},
	volume = {37}, series = {Proceedings of Machine Learning Research},
	address = {Lille, France},
	month = {07--09 Jul},
	publisher = {PMLR},
	pdf = {http://proceedings.mlr.press/v37/xuc15.pdf},
	url = {http://proceedings.mlr.press/v37/xuc15.html},
	abstract = {Inspired by recent work in machine translation and object detection, we introduce an attention based model that automatically learns to describe the content of images. We describe how we can train this model in a deterministic manner using standard backpropagation techniques and stochastically by maximizing a variational lower bound. We also show through visualization how the model is able to automatically learn to fix its gaze on salient objects while generating the corresponding words in the output sequence. We validate the use of attention with state-of-the-art performance on three benchmark datasets: Flickr8k, Flickr30k and MS COCO.}
} 
```
