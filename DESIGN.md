# Design of DVC pipeline for experimental platform
pipeline defined in dvc.yaml
## Prepare
cmd: python src/prepare/prepare.py
params (for now):
    dataset: ...
    category: ...
    num_point: ...
    var_range: ...
    (3DMM preprocessing pipeline in mini modules to reuse some with PointNet)

function definition of prepare.py:
1. read in parameters from params.yaml
2. build a DataPreparer class with given params
    ```
    data_preparer = DataPreparer(params)
    ```
3. save a .pickle version of clean data (potentially a list of Parts) at data/clean
    ```
    data_preparer.preprocess_and_save()
    ```

## Train
cmd: python src/train/train.py
params (for now):
  mode: 'PointNet' # or 'PCA'
  dataset:
    name: 'shapenet'
    category: 'car'
    batch_size: 16
  training:
    pointnet:
      init_lr: 0.0001
      epochs: 301
      decay_step: 30
      decay_rate: 0.7
    #PCA:
      # ...
      # ...
  logging:
    name: 'PointNet_demo'
    pointnet:
      log_step: 10

function definition of train.py:
1. read in parameters from params.yaml
2. build a Model object from params
    ```
    model = Model(params)
    ```
3. train model
    save step logs at /reports
    save trained Model object via pickle or pytorch save model

## Evaluate
TBD...

a few thoughts:
1. metrics: 
    reconstruction loss on unseen data
    - [ ] ... 
2. plots:
    qualitative examination of reconstructed object (use your eyes)
    visualize distribution of anatomy and implants (should have gaps between, but how?)
    - [ ] ... 


## To Do:
 - [ ] abstract DataPreparer class that parse parameters and have an abstract function `preprocess_and_save`
 - [ ] explore pipeline design pattern and apply on DataPreparer
 - [ ] define Mini modules during data preprocessing
 - [ ] abstract Model class that parse parameters and have abstract functions: `train`, `encode`, `decode`
 - [ ] design and train a MLP to map encodings of anatomy to implants


