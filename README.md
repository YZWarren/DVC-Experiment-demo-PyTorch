# DVC experiment with toy Pytorch models

Build environment: `make install`

## Steps to setup dvc
1. created virtual environment
```
conda create -n dvc python=3.9
```

or

```
virtualenv -p python3 .venv
source .venv/bin/activate
```

2. install dvc
```
pip install dvc
```
3. get dvc extension in vscode

## Versioning data in remote storage
1. Set remote storage
```
dvc remote add -d storage <url_to_remote_storage>
```

`-d` means set as default remote. See supported `<url_to_remote_storage>` at <https://man.dvc.org/remote>

2. Start tracking data
```
dvc add data
```

This will create data.dvc file to track and add data into .gitignore, git add & commit data.dvc and .gitignore if want to stage this change

3. Usage
`dvc add data` and `dvc push` if made changes to data
`dvc pull` if need data

## Pipelines
1. [pipeline structure example](https://dvc.org/doc/user-guide/pipelines/defining-pipelines):
```
stages:
  prepare: ... # stage 1 definition
  train: ... # stage 2 definition
  evaluate: ... # stage 3 definition
```
2. `dvc repro` to run pipeline
