# import libraries
import sys

from sqlalchemy import create_engine
import pandas as pd
import re
import pprint

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import warnings
warnings.filterwarnings('ignore')

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """Method to load data from database file.
        
        Args: 
            str: database_filepath
        Returns: 
            dataframes: X, y cleaned data from database file and separated into independent and dependent variables
            list: target names for dependent variables in y dataframe
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df.drop(['message', 'genre', 'original', 'child_alone'], axis=1)
    return X, y, y.columns

def tokenize(text):
    """Method to tokenize from text.
        
        Args: 
            str: text column from dataframe
        Returns: 
            list: cleaned tokens
    """
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    tokens = word_tokenize(str(text))
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Method to build model.
        
        Args: 
            None
        Returns: 
            intialized model
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words={'english'}, tokenizer=tokenize, min_df=0.01))
             ,('tfidf', TfidfTransformer())
             ,('clf', MultiOutputClassifier(LogisticRegression(random_state=0), n_jobs=-1))
            ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """Method to apply and evaluate results from model.
        
        Args: 
            model: trained model
            dataframes: X_test, Y_test
            list: category names
        Returns: 
            dict: results classification reports
    """
    
    Y_pred = model.predict(X_test)
    
    results = {}
    for i in range(len(Y_test.columns)):
        results[category_names[i]] = metrics.classification_report(Y_test.iloc[:,i], Y_pred[:,i])
    return results

def save_model(model, model_filepath):
    """Method to save trained model.
        
        Args: 
            model: trained model
            str: model filepath
        Returns: 
            None
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()