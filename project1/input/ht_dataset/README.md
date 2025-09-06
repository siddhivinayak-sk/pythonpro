---
dataset_info:
  features:
  - name: user_input
    dtype: string
  - name: response
    dtype: string
  - name: target
    dtype: int64
  splits:
  - name: train
    num_bytes: 30520
    num_examples: 50
  download_size: 18403
  dataset_size: 30520
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
