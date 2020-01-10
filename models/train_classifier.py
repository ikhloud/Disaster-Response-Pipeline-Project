import sys
# import libraries
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
import pickle


# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])


def load_data(database_filepath):
    '''
    Load data from database into dataframe
    Input:
        database_filepath: File path of sql database
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Name of categories (36 categories)
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disasters', engine)
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names
    


def tokenize(text):
    '''
    Tokenizes text data
    Input: 
       text : original message text
    Output:
       clean_tokens : cleaned text
    '''
   
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove Stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
def build_model():
    '''
    Build a Machine learning pipeline and Using grid search
    Input: 
        None
    Output:
        cv: Results of GridSearchCV
        '''
    # Build pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))]) 
    
    # Parameters for GridSearch
    parameters =  {'clf__estimator__n_estimators': [50, 100],
                   'clf__estimator__min_samples_split': [2, 3, 4],
                   #'tfidf__use_idf': (True, False),
                   #'clf__estimator__criterion': ['entropy', 'gini']
                    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs = -1 )
    return cv

def evaluate_model(model, X_test, Y_test, category_names):

   '''
    Evaluate model performance using test data, Print accuracy and classfication report
    Input:
        model : the model that need to be evaluated after we trained 
        X_test : test data ( Features )
        Y_test : Label Features for test data 
        category_names : Name of categories (36 categories)  
    Output:
        None
        
   '''

   y_pred = model.predict(X_test) 
    
    # print classification report
   for i in range(len(category_names)):
       print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))

    # print accuracy score
   print("accuracy = {}".format(np.mean(Y_test.values == y_pred)))
    
    
def save_model(model, model_filepath):
    
    '''
    Save model as pickle file
    Input: 
        model : Model to be saved 
        model_filepath : the file path to save the model
    Output: 
        None 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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