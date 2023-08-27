### Table of Contents

1. [Installation](#installation)
2. [File Descriptions](#files)
3. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There is the below libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.10.

<dl>
  <dt>Python Packages used for this project are:</dt>
  <dd>Numpy</dd>
  <dd>Pandas</dd>
  <dd>Scikit-learn</dd>
  <dd>lightGBM</dd>
  <dd>NLTK</dd>
  <dd>SQLAlchemy</dd>
  <dd>Flask</dd>
  <dd>Plotly</dd>
</dl>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions <a name="files"></a>

 app  
 |- template  
 |- master.html  # main page of web app  
 |- go.html  # classification result page of web app  
 |- run.py  # Flask file that runs app  
 data  
 |- disaster_categories.csv  # data to process  
 |- disaster_messages.csv  # data to process  
 |- process_data.py  
 |- InsertDatabaseName.db   # database to save clean data to  
 models  
 |- train_classifier.py  
 |- classifier.pkl  # saved model   

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Appen (formally Figure 8) for the data.  You can find the Licensing for the data and other descriptive information at the Appen link available [here](https://appen.com/).  Otherwise, feel free to use the code here as you would like!
