import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')

# download necessary NLTK data 
import nltk 
nltk.download(['stopwords','punkt']) 

# NLP toolkit
import re 
import pandas as pd 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

# Scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, recall_score, precision_score


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data_msg', engine)
    
    # Preparing model data 
    category_names = [col.split('-')[0] for col in df.categories.str.split(";")[0]]
    X = df.message.values
    y = df[category_names].values
    return X,y, category_names


def build_model():
    # Build the pipeline
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),                 
    ('clf', MultiOutputClassifier(DecisionTreeClassifier(random_state=0)))  
    ])
    param_grid = {
    'clf__estimator__max_depth': [10, 20],  
    'clf__estimator__min_samples_split': [2, 10],
    }

    gcs = GridSearchCV(estimator=pipeline, param_grid=param_grid,verbose=3)
                               
    return gcs


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test) 
    
    dict_metrics = {}
    f1 = 0
    prec = 0
    N= len(category_names) 
    for i in range(N):
        dict_metrics[category_names[i]] = {'f1_score': f1_score(Y_test[:,i], Y_pred[:,i], average= 'weighted'),
                           'precision': precision_score(Y_test[:,i], Y_pred[:,i], average= 'weighted'),
                           'recall': recall_score(Y_test[:,i], Y_pred[:,i], average= 'weighted')}
        f1 += f1_score(Y_test[:,i], Y_pred[:,i], average= 'weighted')
        prec += precision_score(Y_test[:,i], Y_pred[:,i], average= 'weighted')
    
    # Printing results        
    print(f"MSG. f1-avg: {f1/N}", f"prec-avg: {prec/N}")
    
    for k,v in dict_metrics.items():
        print(f"{k}:{v}")


def save_model(model, model_filepath): 
    with open(f"{model_filepath}", 'wb') as file:
        pickle.dump(model, file)

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