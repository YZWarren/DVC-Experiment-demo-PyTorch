# DVC experiment with toy Pytorch models

## Steps to setup dvc
1. created conda environment with newest python
```
conda create -n dvc python=3.10
```
2. install dvc
```
pip install dvc
```
3. get dvc extension in vscode

## Versioning data in remote storage
1. Start tracking data
```
dvc add data
```
    This will create data.dvc file to track and add data into .gitignore, git add & commit data.dvc and .gitignore if want to stage this change

2. Storing data
```
dvc remote add -d storage <url_to_remote_storage>
```

`-d` means set as default remote. See supported `<url_to_remote_storage>` at <https://man.dvc.org/remote>

3. `dvc pull` if need data
