# AI Healthcare API to predict the risk of Suicide attempt from Text

## Description of folders and files

```tree
   ai_healthcare_fastapi
   |-- Dockerfile
   |-- README.md
   |-- api.py
   |-- model_utils.py
   |-- requirements.txt
```   
## Opensource Dataset details

Dataset link :- https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

- class 1 : suicide
- class 2 : non-suicide

## Objective

The problem statement is to develop AI system to predict whether the user is going to attempt suicide or not based on text.
 
## Requiremnts to execute API :-

Download this file :- https://github.com/google-research/bert/blob/master/tokenization.py

## Execute API

```shell
python3 -m uvicorn api:app --reload
```
