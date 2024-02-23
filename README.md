# CascadeXML
Pytorch implementation of  CascadeXML: Rethinking Transformers for End-to-end Multi-resolution Training in Extreme Multi-label Classification.

## Dataset
CascadeXML is tested on datasets in [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). `Amazon-670K`, `Amazon-3M`, `Wiki-500K`, `AmazonCat-13K` and `Wiki10-31K` are supported. Before training, the datasets have to be prepared and label clusters have to be created. We use Eclare style clusters for CascadeXML.

## Setup Environment
Run the below command, this will create a new conda environment named cascadexml with all the dependencies required to run CascadeXML.
```
conda create -n cascadexml python==3.10
conda activate cascadexml
bash install.sh

```
## Training
Prepare the dataset and clusters and run the src/main.py
An example command to train CascadeXML on `Wiki10-31K` dataset on a single GPU is provided below:
``` 
python src/main.py --num_epochs 15 --dataset Wiki10-31K --batch_size 64 --max_len 256 --mn Cascade_Wiki10-31K --topk 64 64 --cluster_name Eclusters_54.pkl --rw_loss 1 1 1  --embed_drops 0.3 0.3 0.4 --warmup 2 
```
Note: Large datasets like `Amazon-3M` may not fit on a single GPU. For such datasets we recommend enabling the `--sparse` flag. This flag uses a SparseAdam optimizer instead of standard AdamW and helps reduce memory costs. Unfortunately sparse gradients are not supported by nvcc so DDP cannot be used in this mode. Upto 2 GPUs can be used in sparse mode for model parallel.

### Other hyperparams
Amazon-670K
```
python src/main.py --lr 1e-4 --num_epochs 15 --dataset Amazon-670K --batch_size 64 --max_len 128 --mn CascadeXML_A670K_2111 --topk 128 256 400 --cluster_name Eclusters_1865.pkl --embed_drops 0.2 0.25 0.4 0.5
```
Wiki-500K
```
python src/main.py --lr 1e-4 --num_epochs 15 --dataset Wiki-500K --batch_size 64 --max_len 128 --mn CascadeXML_Wiki500K_2111 --topk 128 256 512 --cluster_name Eclusters_1865.pkl --embed_drops 0.2 0.25 0.35 0.5
```



## Citation

If you find this repository useful, please consider giving a star and citing our paper:

```
@article{Kharbanda2022CascadeXMLRT,
  title={CascadeXML: Rethinking Transformers for End-to-end Multi-resolution Training in Extreme Multi-label Classification},
  author={Siddhant Kharbanda and Atmadeep Banerjee and Erik Schultheis and Rohit Babbar},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.00640}
}
```
