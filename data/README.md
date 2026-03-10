# Data Directory

This directory should contain the dataset files for D1 and D2.

## Directory Structure

```
data/
├── D1/
│   ├── cases/
│   │   ├── cases.csv
│   │   └── ad_cases.pkl
│   ├── hash_info/
│   │   ├── node_hash.pkl
│   │   ├── type_hash.pkl
│   │   ├── type_dict.pkl
│   │   └── channel_dict.pkl
│   └── samples/
│       ├── train_samples.pkl
│       └── test_samples.pkl
└── D2/
    ├── cases/
    │   ├── cases.csv
    │   └── ad_cases.pkl
    ├── hash_info/
    │   ├── node_hash.pkl
    │   ├── type_hash.pkl
    │   ├── type_dict.pkl
    │   └── channel_dict.pkl
    └── samples/
        ├── train_samples.pkl
        └── test_samples.pkl
```

## File Descriptions

- `cases.csv`: Case information file
- `ad_cases.pkl`: Anomaly detection case labels (pickle format)
- `node_hash.pkl`, `type_hash.pkl`, `type_dict.pkl`, `channel_dict.pkl`: Hash/dictionary files
- `train_samples.pkl`, `test_samples.pkl`: Training and testing samples

## Note

Due to data privacy and size considerations, sample data files are not included in this repository. Please prepare your own data files according to the above structure.
