Use Python 3.9.6

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


How to download PhysioNet nurse note data
```
mkdir -p downloaded
cd downloaded
wget -r -N -c -np --user lixinw2 --ask-password https://physionet.org/files/deidentifiedmedicaltext/1.0/

```


How to download DocRED data (optional)
Follow instructions here
https://github.com/thunlp/DocRED/tree/master/data

```
./downloaded/DocRED_baseline_metadata
./downloaded/DocRED_baseline_metadata/rel2id.json
./downloaded/DocRED_baseline_metadata/word2id.json
./downloaded/DocRED_baseline_metadata/char_vec.npy
./downloaded/DocRED_baseline_metadata/char2id.json
./downloaded/DocRED_baseline_metadata/vec.npy
./downloaded/DocRED_baseline_metadata/ner2id.json

```


