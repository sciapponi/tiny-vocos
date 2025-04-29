# tiny-vocos

To train:

create a conda environment with python 3.11:

```shell
conda create -n tinyvocos python=3.11
conda activate tinyvocos
```

Install the requirements:
```shell
pip install -r requirements.txt
pip install -r requirements-train.txt
```

If fairseq returns errors fix it with this install (it might take a while):
```shell
pip install git+https://github.com/One-sixth/fairseq.git
```
