# DiaMaT
An MT diagnostics tool. DiaMaT learns from corpora containing millions of translations but offers explanations on sentence level. 

![screenshot](https://github.com/DFKI-NLP/diamat/blob/master/resources/screenshot.png)

Code accompanying the paper:

```
@inproceedings{Schwarzenberg_tse_2019,
  title = {Train, Sort, Explain: Learning to Diagnose Translation Models},
  booktitle = {Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT): Demonstrations.},
  author = {Schwarzenberg, Robert and Harbecke, David and Macketanz, Vivien and Avramidis, Eleftherios and M\"oller, Sebastian},
  location = {Minneapolis, Minnesota, USA},
  year = {2019}
  }
```

## Info 
A DiaMaT demo is hosted [here](http://diamat.dfki.de). Optimized for the Firefox browser w/ a resolution of 1920x1080.

DiaMaT deploys the [iNNvestigate](https://github.com/albermax/innvestigate) toolbox.

To facilitate the replication of experiments, if this repo is cloned, 500 MB of data will be directly downloaded from the GitHub LFS server.

## Installation 
Download embeddings:
```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
```
Link wiki.de.vec and wiki.en.vec in ./data/input/word_vectors (see ./data/input/config.INI)

Use Python 3.6, e.g.:
```
conda create --name diamat python=3.6
```
Activate environment:
```
source activate diamat
```

Install requirements (if CUDA GPU is available install tensorflow-gpu==1.7.0):

``` 
pip install -r requirements.txt
```

Install SpaCy language models

```bash
python -m spacy download de
python -m spacy download en
```

## Run / Replicate Experiments
Please, first validate data/input/config.INI.

Unzip ./data/output.zip.

Run train.py and then explain.py, afterwards link data/output/explain.jsonl in server/static/input, e.g.:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py 2>&1 | tee -a ./data/output/fit_gpu_deeplee_quadro.log &&
CUDA_VISIBLE_DEVICES=0 python3 explain.py 2>&1 | tee -a ./data/output/fit_gpu_deeplee_quadro.log &&
ln -s ../../../data/output/explain.jsonl ./server/static/input/explain.jsonl 2>&1 | tee -a ./data/output/fit_local.log
```
Then start the server:
```
cd server &&
sh run_flask.sh
```
Visit 
```
firefox localhost:5000
```

## Data 
All preprocessed data needed to replicate the experiments is contained in ./data/output
- ./data/output/train.jsonl > 1M JSON lines to train the text classifier
- ./data/output/val.jsonl > 100k JSON lines to validate the classifier during training
- ./data/output/xtrain.jsonl > 100k JSON lines to train the explainability method (if needed)
- ./data/output/explain.jsonl > ca 20k JSON lines to test DiaMaT on (drawn from the official WMT test sets / excluding WMT13)

./data/output/explain.jsonl contains contributions (from the experiments) which are overwritten by explain.py 

### How to use your own data: 
- Prepare three parallel text files: source.txt, machine.txt and human.txt
- Then run:
```
python preprocess.py
```
Afterwards ./data/output/bundle.jsonl contains the bundled texts, tokens, indices etc. Rename bundle.jsonl, replace source.txt, machine.txt, human.txt and repeat the step if you would like to preprocess more data.

Update ./data/input/config.INI and run train.py and explain.py again. 

Update server/static/input/explain.jsonl w/ the new explanations and run the server again.
