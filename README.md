# Project Generative Model for Stock Prediction with Market News

<p align="center">
  <a href="#tutorials">Tutorials</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
</p>


## Installation


0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

1. At first we need to download [dataset](https://huggingface.co/datasets/Zihan1004/FNSPID), it will take about 25GB of space on your device, run following command:

```bash
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv
```

2. Then we need to preprocess data, set the fields stocks_dir and path_to_news with paths to "full_hsitory/" and "nasdaq_exteral_data.csv" respectively, run the following command (you will need accelerator for this, it will take about 3 hours on GPU A100):

```bash
python3 preprocess_data.py
```

Now, there's a directory "saved/preprocessed/" in our project, where all required data is located, you may delete downloaded files.

3. To train a model, run the following command:

```bash
python3 train.py
```
You may change hyperparameters for training in "src/configs/diff_model.yaml".

4. To run inference (evaluate the model or save predictions):

```bash
python3 inference.py 
```

You may change hyperparameters for inferencing in "src/configs/inference.yaml".

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
