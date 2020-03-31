# Multimodal Transformer for Unaligned Multimodal Language Sequences
Unofficial pytorch implementation for [Multimodal Transformer](https://arxiv.org/pdf/1906.00295.pdf)
- This code is only for IEMOCAP dataset
- Some part of code are adapted from [official code of authors](https://github.com/yaohungt/Multimodal-Transformer) and [fairseq repo](https://github.com/pytorch/fairseq)
- Datasets are available in author's github page above

## Requirements
- python 3.7.4   
- pytorch 1.4     
- cuda 10.1  
- fairseq  
- sklearn  


## Run  
for linux shellscripts   
```
$ bash scripts/train.sh  
```


## Experiments  
Hyperparameters reported on papaer are used     
   
**Reproduction results**
|              | Happy(acc) | Happy(f1) | Sad(acc) | Sad(f1) | Angry(acc) | Angry(f1) | Neutral(acc) | Neutral(f1) |
| :----------: | :--------: | :-------: | :------: | :-----: | :--------: | :-------: | :----------: | :---------: |
| IEMOCAP test |    85.6    |   79.0    |   79.4   |  70.3   |    75.8    |   65.4    |     59.2     |    44.0     |


## Reference   
- Multimodal Transformer for Unaligned Multimodal Language Sequences : https://arxiv.org/pdf/1906.00295.pdf  
- Official Pytorch Implementation : https://github.com/yaohungt/Multimodal-Transformer   
  - Official Fairseq Repository : https://github.com/pytorch/fairseq   