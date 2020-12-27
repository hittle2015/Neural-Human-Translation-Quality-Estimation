# Neural-Human-Translation-Quality-Estimation
Deep Learning HTQE models hacked from 11 SOTA, such as TextCNN1D, RCNN, DPCNN, Transformers, neural architectures.
## How to Use
###1. Configuration：

	```
	pip install -r requirements.txt
	```

###2. Confirm the nature of task: regression or classification：

	1. classification：classify translation into 'good' (1) or 'poor' (0)
	2. regression: quantitative score translation per certain scale, e.g. percent scale, within the range 0-1, etc.


###3. prepare your data as jsonl format and wordembedding：
   run dataPreprocess.py to generate train, valid and test jsonl file as well as wordembedding matrix

	```
{"id": 10750, "text": "[CLS] The Troika’s claims on Greece need not be reduced in face value, but their maturity would have to be lengthened by another decade, and the interest on it reduced. Further haircuts on private claims would also be needed, starting with a moratorium on interest payments.[SEP] 希腊还必须再次重组并削减公债。三驾马车的希腊债权不必在面值上有所让步，但期限必须延长十年，利息也必须减少。此外，还需要私人债权的进一步“剃头”，可以从暂停利息支付开始入手。[SEP]\n", "labels": [1]}
{"id": 2086, "text": "[CLS] Weaker forts included Fort Larned, originally built mainly of adobe. [SEP] 较弱的堡垒包括拉恩堡, 它最初主要是用土豆建造的. [SEP]\n", "labels": [0]}
	```
  

###4. build your own dataloader

###5. change the configuration files wherever is necessary

###6. run train.py

	```
	python train.py
	```

## features
- [x] tensorboard visualization (metrics, )
- [x] transformers integrated [huggingface/transformers](https://github.com/huggingface/transformers)
- [x] regression and classification compatible
- [x] support multi-GPU
## Acknowledgement
This repo(https://github.com/jeffery0628/text_classification/) has saved me a lot of trouble of coding from scratch.  
