# Disaster Response Pipeline Project

## Project Overview
In this project, I'll apply data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages ( I used a Machine Learning Pipeline to build a supervised learning model ).
Figure Eight are providing pre-labeled tweets and messages from real-life disasters. 

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


## Project Components
There are three components for this project.

#### 1. ETL Pipeline

File  process_data.py, contains a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

#### 2. ML Pipeline

File train_classifier.py, contains a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

I create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification)

#### 3. Flask Web App

File run.py, contains a Flask file
Add data visualizations using Plotly in the web app. 
I create data visualizations in my web app based on data you extract from the SQLite database.


## Here's the file structure of the project:

#### web app
- app

| - template

| |- master.html  # main page of web app

| |- go.html  # classification result page of web app

|- run.py  # Flask file that runs app

#### ETL pipeline
- data

| - disaster_categories.csv  # data to process 

| - disaster_messages.csv  # data to process

| - process_data.py :

| - InsertDatabaseName.db   # database to save clean data to

#### ML pipeline
- models

|- train_classifier.py

|- classifier.pkl  # saved model 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

##### To find your enviroment of your workspace

1- open new terminal and print :
env|grep WORK

  this will print:
  WORKSPACEDOMAIN=udacity-student-workspaces.com
  WORKSPACEID=view6914b2f4

2- identify your website link by doing substitution :
https://WORKSPACEID-3001.WORKSPACEDOMAIN


