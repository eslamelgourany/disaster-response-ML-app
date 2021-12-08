# import libraries
import re
import sys
import pandas as pd
import numpy as np
import pickle
import os
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    """
    Function used to load the database and table.
    Inputs:
        database_filepath: is the path for the database '.db'
    Returns: None
    """
    engine = create_engine('sqlite:///' +str(database_filepath))
    table_name = os.path.basename(database_filepath).split('.')[0]
    
    # Read SQL table
    df = pd.read_sql_table(table_name, con=engine)
    
    # Define X and Y for the model.
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    column_names = Y.columns
    
    return X, Y, column_names

def tokenize(text):
    """
    Function to clean the input text.
    Input: String of text.
    Returns: List of cleaned words.
    """
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    # Tokenize
    tokens = word_tokenize(text) 
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer() 
    
    # Clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok) 
        
    # Remove stop words
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return clean_tokens

def build_model():
    """
    Function used to create the ML pipeline and parameters to validate.
    Returns: Model Object.
    """
    # Model Pipeline.
    pipeline = Pipeline([
    
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Different parameters to validate.
    parameters = {
    'clf__estimator__n_estimators': [20, 50],
    'clf__estimator__min_samples_split': [2, 3, 4],  
    'tfidf__use_idf': (True, False),
    }
    
    # Grid search the parameters.
    scorer = make_scorer(f1_score, average = 'weighted')
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function used to evaluate model on the testing set.
    Inputs:
        model: Model output from the build model function
        X_test: Testing features
        Y_test: Testing dataset
        category_names: Y names used to iterate through and calculate the accuracy.
    """
    # predict
    y_pred = model.predict(X_test)
    
    # Score
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Function to save model.
    Inputs: 
        model: Model to be saved.
        model_filepath: Path to the model file
    Returns: None
    """
    # Save model to pkl file.
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main function to call the above functions.
    """
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