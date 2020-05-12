# Targeted Sentiment Analysis
## IN5550 ‒ Home Exam Project
### University of Oslo ‒ Spring 2020

This repository accompanies the written report. In addition to my own code, it contains pre-code written by IN5550 course lecturers.

### The task in short 
*From the task description:* Fine-grained Sentiment Analysis (SA), sometimes referred to as Opinion Analysis/Mining, is the task of identifying opinions in text and analyzing them in terms of their polar expressions, targets, and holders. In this task we will focus on targeted SA, i.e. the identification of the target of opinion along with the polarity with which it is associated in the text (positive/negative). In the example below, for instance, the target of the opinion is disken ‘the disk’ and it is ascribed a positive polarity by the surrounding context.

| | 𝘗𝘖𝘚 | | | |  
|:--|:--|:--|:--|:--|  
|*Denne*|**disken**|*er*|*svært*|*stillegående*|
|This   |disk      |is  |very   |quiet-going   |  

### Data format
The dataset used is [NoReC-fine](https://github.com/ltgoslo/norec_fine), a dataset for fine-grained sentiment analysis in Norwegian. It was converted from JSON to a stripped-down version of ConLL-U for this task, and given as part of the pre-code. The labels follow the [BIO / IOB2 format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging), with added polarity, and there are a total of five labels:  

- B-targ-Positive
- I-targ-Positive
- B-targ-Negative
- I-targ-Negative
- O  

##### An example of a tagged sentence.
   
*# sent_id = 501595-13-04* 

|Token              |Tag            |
|:--                |:--            |
|Munken             |B-targ-Positive|
|Bistro             |I-targ-Positive|
|er                 |O              |
|en                 |O              |
|hyggelig           |O              |
|nabolagsrestaurant |O              |
|for                |O              |
|hverdagslige       |O              |
|og                 |O              |
|uformelle          |O              |
|anledninger        |O              |
|.                  |O              |

### Modelling
The main objective of the home exam is to train a neural system to perform targeted sentiment analysis for Norwegian text.

#### Baseline model
A simple BiLSTM-model.

#### Experimental evaluation
Evaluate the effect of at least three different changes to the basis system. The evaluation of changes to the system should be performed on the development set.

#### Held-out testing
The best configuration of your system following experimentation should be evaluated on the test set.

### Evaluation
The models will be evaluated on two different metrics: [proportional F1 and binary F1](https://en.wikipedia.org/wiki/F1_score). *Binary Overlap* counts any overlapping predicted and gold span as correct. *Proportional Overlap* instead assigns precision as the ratio of overlap with the predicted span and recall as the ratio of overlap with the gold span, which reduces to token-level F1. Proportional F1 is therefore a stricter measure than Binary F1.

### Possible directions for experimentation
1. Experimenting with alternative label encoding (e.g. BIOUL)
2. Compare *pipeline* vs. *joint prediction* approaches.
3. Impact of different architectures:
	- LSTM vs. GRU vs. Transformer
	- Include character-level information
	- Depth of model (2-layer, 3-layer, etc)
4. 
Effect of using pretrained models ([ELMo](https://github.com/HIT-SCIR/ELMoForManyLangs), [BERT](https://github.com/botxo/nordic_bert), or [Multilingual Bert](https://github.com/google-research/bert/blob/master/multilingual.md))
5. A small error analysis (confusion matrix, the most common errors).  

------------
Project Organisation 
------------


    ├── LICENSE              <- Not added yet. Reason: exam 
    ├── README.md            <- The top-level README for developers using this project.
    │
    ├── babil                <- Source code for use in this project.
    │   │
    │   ├── __init__.py      <- Makes babil a Python module. Babil approves of this.
    │   ├── __main__.py      <- Makes babil callable. Hello!
    │   │
    │   ├── coleus           <- Word embeddings model trainer.
    │   │   ├── config.py    <- Argument parsing and configurations for coleus.
    │   │   ├── helpers.py   <- Helper methods for coleus.
    │   │   └── model.py     <- The class containing the model. 
    │   │
    │   ├── data             <- Scripts to download or generate data.
    │   │   └── preprocessing.py
    │   │
    │   ├── features         <- Scripts to turn raw data into features for modeling.
    │   │   └── embeddings.py
    │   │
    │   ├── models           <- Scripts to train models and then use trained models 
    │   │   │                   to make predictions.
    │   │   ├── baseline.py  < 
    │   │   ├── fasttext.py
    │   │   ├── improved.py
    │   │   └── run.py
    │   │
    │   └──  utils            <- Scripts containing helper functions 
    │       └── config.py     <- Argument parsing and configurations for babil.
    │       ├── helpers.py    <- Helper methods for coleus.
    │       └── metrics.py    <- Model metrics class.
    ├── data
    │   ├── interim          <- Intermediate data that has been transformed.
    │   ├── processed        <- The final, canonical data sets for modeling.                
    │   └── raw              <- The original, immutable data dump.
    │
    ├── embeddings           <- Smaller embeddings for running tests locally.
    │
    ├── figures              <- Anything related to data visualisation.
    │
    ├── logs                 <- Log files.
    │
    ├── misc                 <- If it fits, it sits... Or something along those lines.
    │
    ├── models               <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks            <- Jupyter notebooks.
    │
    ├── references           <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures          <- Generated graphics and figures to be used in reporting.
    │
    ├── environment.yml      <- The environment file for reproducing the analysis enviroment using Conda.             
    │
    ├── requirements.txt     <- The requirements file for reproducing the analysis environment, e.g.
    │                           generated with `pip freeze > requirements.txt`.
    │
    ├── path_config.json     <- Keeps track of the various paths used to load or save projects.
    │
    ├── conllu_parser.sh     <- Script for parsing CoNNL-U files, in batch, to one sentence per line.
    │
    ├── setup.py             <- makes project pip installable (pip install -e .) so src can be imported.
    │        
    └── SEED.txt             <- Much seed. Very strong. Wow.

--------
