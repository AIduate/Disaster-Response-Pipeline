# Disaster-Response-Pipeline

The purpose of this project is to create a pipeline of data provided by Figure 8 to Udacity to be combined and cleaned and fed into a machine learning pipeline that will use nature language processing techniques to classify messages related to disasters to the most likely appropriate response.

## Table of contents
* [Results](#results)
* [Acknowledgements](#acknowledgements)
* [Technologies](#technologies)
* [Libraries](#libraries)
* [Files](#files)

# Results
* Most messages are just 'related' and there were 0 messages for 'child alone' for the model to train on.
* In the future, I plan to check whether it would be helpful to combat the class imbalance with a couple techniques like upsampling, stratified train test split, etc.
Would also like to try optimizing hyper parameters a bit more in depth.

<img width="1133" alt="Untitled" src="https://user-images.githubusercontent.com/33467922/127799474-a89fafe0-debd-44d3-ae01-a8c09ca38982.png">


# Acknowledgements
* Udacity
* Figure 8

# Technologies
* Python 3.6
* Html
* Javascript

# Libraries

```
from sqlalchemy import create_engine
import pandas as pd
import re
from sklearn
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])
```

```
https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/
https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/
https://cdn.plot.ly/plotly-latest
```

# Files
- app

 - template

    - master.html  - main page of web app

    - go.html  - classification result page of web app

 - run.py  - Flask file that runs app


- data

  - disaster_categories.csv  - data to process 

  - disaster_messages.csv  - data to process

  - process_data.py - data processing/cleaning python script

  - InsertDatabaseName.db   - database to save clean data to


- models

   - train_classifier.py - train and save classifier
   - classifier.pkl  - saved model 


