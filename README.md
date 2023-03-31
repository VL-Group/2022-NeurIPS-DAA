## Introduction
PyTorch implementation for NeurIPS2022 paper of [**“A Differentiable Semantic Metric Approximation in Probabilistic Embedding for Cross-Modal Retrieval”**](https://openreview.net/forum?id=-KPNRZ8i0ag).  It is built on top of the [SGRAF](https://github.com/Paranioar/SGRAF) in PyTorch. 

![image](assets/Motivation.png)
## Requirements and Installation
We recommended the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) 1.10.0
* [NumPy](http://www.numpy.org/) (>1.19.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Pretrained model
If you don't want to train from scratch, you can download the pretrained model from [here](https://drive.google.com/file/d/1hCiXyQBrYF7eP7JtNTaHQoqSZ18-mb2X/view?usp=sharing) (**SGR** for MS-COCO model), [here](https://drive.google.com/file/d/1o-Wch7pJMwOyf-RqEvsXgAUBhIO9fv8g/view?usp=sharing) (**SAF** for MS-COCO model), [here](https://drive.google.com/file/d/1Q0Ttw4yViWnupJA1whTAYGYHr-O56D4L/view?usp=sharing) (**SGR** for Flickr30K model) and [here](https://drive.google.com/file/d/1hvixWqDDCbkbBYK28dyZeEI6Nlc0qfAx/view?usp=sharing) (**SAF** for Flickr30K model). The performance of these pretrained models are as follows:

![image](assets/Result-MSCOCO.png)

## Prepare data
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies, which can be downloaded by using:

```bash
wget https://iudata.blob.core.windows.net/scan/data.zip
wget https://iudata.blob.core.windows.net/scan/vocab.zip
```
Another download link is available below：

```bash
https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC
```

To speed up dataset loading, we convert these features from numpy.array to HDF5 file. Modify the **data_path** in `np2h.py` and then run `np2h.py`:
```bash
python np2h.py
```

## Training
Modify the **data_path**, **vocab_path**, **model_name**, **logger_name** in the `opts.py` file. Then run `train.py`:

For MSCOCO:

```bash
(For SGR) python train.py --data_name coco_precomp --num_epochs 30 --learning_rate 0.00015 --lr_update 20 --world_size 4 --module_name SGR --daa_weight 25
(For SAF) python train.py --data_name coco_precomp --num_epochs 30 --learning_rate 0.00015 --lr_update 20 --world_size 4 --module_name SAF --daa_weight 25
```

For Flickr30K:

```bash
(For SGR) python train.py --data_name f30k_precomp --num_epochs 40 --learning_rate 0.0006 --lr_update 30 --world_size 1 --module_name SGR --daa_weight 10
(For SAF) python train.py --data_name f30k_precomp --num_epochs 40 --learning_rate 0.0006 --lr_update 20 --world_size 1 --module_name SAF --daa_weight 10
```

## Evaluation

### Test on MSCOCO
To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using `--data_name coco_precomp`.
```bash
python evaluation.py
```


### Test on Flickr30K
To test on Flickr30K, pass `fold5=False` with a model trained using `--data_name f30k_precomp`.
```bash
python evaluation.py
```


## Reference

If you found this code useful, please cite the following paper:
```
@inproceedings{li2022differentiable,
  author    = {Hao Li and
               Jingkuan Song and
               Lianli Gao and
               Pengpeng Zeng and
               Haonan Zhang and
               Gongfu Li},
  title     = {A Differentiable Semantic Metric Approximation in Probabilistic Embedding for Cross-Modal Retrieval},
  Booktitle = {NeurIPS},
  year      = {2022}
}
```

