# Text classification (CS-433, Project 2, 2020)

## Authors: (Team rojlet_lML)

- Mahdi Ben Hassen
- Ahmed Ben Haj Yahia
- Nour Ghribi 

---

This project is about sentiment analysis of tweets. We take a look at binary classification of tweets, that is if each tweet contained a :) or :( emoji.
We used different types of data preprocessing and took advantage of the nltk package. [Natural Language Toolkit](https://www.nltk.org/).
As a baseline model we tried to fit different linear models, we used Naive Bayes, Logistic Regression and SVM. SVM performed the best out of the 3.
Diving into the vast world of NLP, we discovered the different embedding models. GloVe, ELMo, BERT and their improved variatons and types like XLM_RoBERTa which is a multilanguage embedding model based on the BERT one and how we can combine ELMo + GloVe embeddings. More details are covered in the report.

The [BERT model](Notebooks/Bert.ipynb) outperformed all other models with a validation accuracy of 0.8845 and an Accurcy of 0.883 and F1-Score of 0.883 on [AiCrowd](https://www.aicrowd.com/challenges/epfl-ml-text-classification/submissions/109997).

You'll need to install all the requirements in `requirements.txt`using `pip install -r requirements.txt`. Further informations about the different libraries we used can be found in Libraries used.txt.

The best submission can be reproduced using `python3 run.py -f <input.txt> -o <output.csv>` with input.txt and output.csv both given in relative paths. The shell prompt will ask if you want to use the baseline model **SVM** or the best model **BERT**. 

Below we describe how our scripts, notebooks and python files are placed. The file `run.py` will output our best submission.
In addition, there is further details in each notebook, as some require high computaion power and were run on Google Colab or Google Cloud DeepLearning VMs.


All our data and code should be in a directory following this tree: (Data might be not in the github repo but can be downloaded [here](https://go.epfl.ch/rojlet_lML_data)  )


---------
    ├── run.py
    ├── cleaning
    │   ├── data_cleaning.py
    │   ├── potts_tokenizer.py
    ├── scripts
    │   ├── helpers_test.py
    │   ├── helpers_train.py
    ├── models
    │   ├── best_model_implementation.py
    ├── Notebooks
    │   │  ├── Bert.ipynb
    │   │  ├── XLM_roBERTa.ipynb
    │   │  ├── RoBERTa.ipynb
    │   │  ├── Elmo.ipynb
    │   │  ├── Elmo+GLOVE.ipynb
    │   │  ├── Linear_models.ipynb
    │   │  ├── ALL GLOVE MODEL.ipynb
    ├── Resources
    │   ├── glove.6B.300d.txt
    │   ├── slang_dict.txt
    │   ├── spelling_dict1.txt
    │   ├── spelling_dict2.txt
    │   ├── SVM_fit.joblib
    ├── glove_embedding
    │   ├── cleaned_embeddings
    │   │   ├── Full 
    │   │   ├── Sample 
    │   ├── sample_emebeddings.npy
    │   ├── embeddings.pkl
    ├── cleaned_data
    │   ├── cleaned_test_data.txt
    │   ├── cleaned_train_neg_full.txt
    │   ├── cleaned_train_neg.txt
    │   ├── cleaned_train_pos_full.txt
    │   ├── cleaned_train_pos.txt
    ├── Data
    │   ├── test_data.txt
    │   ├── train_neg_full.txt
    │   ├── train_neg.txt
    │   ├── train_pos_full.txt
    │   ├── train_pos.txt
    ├── NN_WEIGHTS
    

---------
