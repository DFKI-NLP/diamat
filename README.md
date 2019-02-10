# DiaMaT
A diagnostics tool. 

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
Unzip ./data/output.zip

Run train.py and then explain.py, afterwards link data/output/explain.jsonl in server/static/input, e.g.:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py 2>&1 | tee -a ./data/output/fit_gpu_deeplee_quadro.log &&
CUDA_VISIBLE_DEVICES=0 python3 explain.py 2>&1 | tee -a ./data/output/fit_gpu_deeplee_quadro.log &&
ln -s ../../../data/output/explain.jsonl ./server/static/input/explain.jsonl 2>&1 | tee -a ./data/output/fit_local.log
```
Then start the server:
```
cd server &&
sh run_server.sh
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
Afterwards ./data/output/bundle.jsonl contains the bundled texts, tokens, indices etc. Rename bundle.jsonl and replace source.txt, machine.txt, human.txt and repeat the step if you would like to preprocess more data.
 