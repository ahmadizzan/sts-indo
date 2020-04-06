# STS Indonesia

## Setup

```
$ pipenv shell
$ pipenv install
```

## Dataset
Translated english dataset from STS task 2012-2016 (12k rows).

## How to train

Sample

```shell
$ python3 src/train_model.py \
	--outputmodelname test.h5 \
	--n_epochs 50 \
	--enc_dim 100
```

For more complete param list, see `src/train_model.py`.
