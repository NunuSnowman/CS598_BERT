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
wget -r -N -c -np --user ${user_name} --ask-password https://physionet.org/files/deidentifiedmedicaltext/1.0/

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


Once python virtual environment is activated

simply run those process one by one
```
python process_1_physionet_nurse_data.py
python process_1_DocRED_data.py
python process_2_validation_data.py
python process_2_validation_model.py
python process_3_train.py
python process_4_test.py
python process_5_run_multiple_combo.py
python process_6_run_cross_dataset.py
python process_7_extension_epoches.py
```

