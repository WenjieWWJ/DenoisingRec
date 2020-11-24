# DenoisingRec


Adaptive Denoising Training for Recommendation.

This is the pytorch implementation of our paper at WSDM 2021:

> [Denoising Implicit Feedback for Recommendation.](https://arxiv.org/abs/2006.04153)
> Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, Tat-Seng Chua.

## Environment
- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4 

For others, please refer to the file env.yaml.

## Usage

### Training
#### T_CE
```
python main.py --dataset=$1 --model=$2 --drop_rate=$3 --num_gradual=$4 --gpu=$5
```
or use run.sh
```
sh run.sh dataset model drop_rate num_gradual gpu_id
```
The output will be in the ./log/xxx folder.

#### R_CE
```
sh run.sh dataset model alpha gpu_id
```
### Inference
We provide the code to inference based on the well-trained model parameters.
```
python inference.py --dataset=$1 --model=$2 --drop_rate=$3 --num_gradual=$4 --gpu=$5
```
### Examples
1. Train GMF by T_CE on Yelp:
```
python main.py --dataset=yelp --model=GMF --drop_rate=0.1 --num_gradual=30000 --gpu=0
```
2. Train NeuMF by R_CE on Amazon_book
```
python main.py --dataset=yelp --model=NeuMF-end --alpha=_0.25 --gpu=0
```
We release all training logs in ./log folder. The hyperparameter settings can be found in the log file. 
The well-trained parameter files are too big to upload to Github. I will upload to drives later and share it here.

## Note  
If you use our code, please kindly cite:

```
@article{wang2020denoising,
  title={Denoising Implicit Feedback for Recommendation},
  author={Wang, Wenjie and Feng, Fuli and He, Xiangnan and Nie, Liqiang and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2006.04153},
  year={2020}
}
```
## Acknowledgment

Thanks to the NCF implementation:
- [Tensorflow Version](https://github.com/hexiangnan/neural_collaborative_filtering) from Xiangnan He. 
- [Torch Version](https://github.com/guoyang9/NCF) from Yangyang Guo.

## License

NUS © [NExT++](https://nextcenter.org/)
