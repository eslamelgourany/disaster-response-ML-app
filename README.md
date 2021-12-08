# Disaster response App

<div id="top"></div>

#### Table of Contents

1. [About The Project](#about-the-project)
2. [Project Motivation](#motivation)
3. [Prerequisites](#Prerequisites)
4. [How to run the app](#Application)
5. [File Descriptions](#files)
6. [License](#License)
7. [Contact](#Contact)
8. [Acknowledgments](#Acknowledgments)


<!-- ABOUT THE PROJECT -->
## About The Project

1. This is a project which is created to demonstrate my Machine Learning skills. It is a project that uses several technologies in the datascience field, starting from the backend development in which Machine learning pipeline is designed and implemented, up to the front end to host the model to be used as application during natural disasters.

2. The data provided are pre-labeled tweets and text messages from real life disasters. These data are prepared using <b>ETL Pipeline</b> and then <b>Machine learning pipeline</b> is used to build a supervised model which learned from the cleaned data.


<!-- MOTIVATION -->
## Project Motivation <a name="motivation"></a>

Typically disaster response organizations get millions of communications, within every thousand messages, few might be relevant to the disaster response professionals. Here comes the AI role to be able to filter, and cateogirze the messages.

<!-- TOOLS -->

## Prerequisites <a name="Prerequisites"></a>

This section list any major frameworks/libraries used to complete the project:

* [Python](https://python.org/)
* [Scikit Learn](https://scikit-learn.org/)
* [NLTK](https://www.nltk.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [SQL Alchemy](https://www.sqlalchemy.org/)
* [Flask](https://www.fullstackpython.com/flask.html)
* [Plotly](https://plotly.com/python/)
* [Requests](https://docs.python-requests.org/en/latest/)
* [BootStrap](https://getbootstrap.com/)
* [HTML](https://html.com/)


<!-- APPLICATION -->
## How to run the app <a name="Application"></a>

1. Clone the repositiry and pip install `requirement.txt`

2. Run the ETL pipeline that cleans data and stores in database python.
Using: `data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

3. Run ML pipeline that trains classifier and saves it.
Using: `python models/train_classifier.py data/DisasterResponse.db models/model.joblib`

4. Run the server and as well render the HTML pages
Using: python `run.py` 


5. Go to http://0.0.0.0:3001/

<!-- FILES -->  
## File Descriptions <a name="files"></a>

![alt text](https://github.com/eslamelgourany/disaster-response-ML-app/blob/main/data/structure.png)

* `run.py`: It is the main file for rendering the HTML pages.
* `contact_me.html`: It has the html code for my contact_me page.
* `go.html`: It has the design for the giving the output of the classification.
* `master.html`: It has the design to render the main page.
* `DisasterResponse.db`: Database which has the table for the cleaned data
* `disaster_categories.csv`: Data that has the categories variants.
* `disaster_messages.csv`: Table that has the messages that will be used as training and validation set.
* `process_data.py`: It is the main file used for cleaning the two csv files. (ETL strategy is applied here)
* `classifier.pkl`: Pickle file that holds the classifier implemented.
* `train_classifier`: The script that has the features and parameters used for designing the ML model.


<!-- LICENSE -->

## License <a name="License"></a>

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
## Contact <a name="Contact"></a>

[@Eslam Elgourany](https://www.linkedin.com/in/eslam-elgourany-75b346111) - eslamelgourany@hotmail.com


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments <a name="Acknowledgments"></a>
I find it very helpful and useful to provide such nice data and information to the public.

* Thank you [Figure eight](https://en.wikipedia.org/wiki/Figure_Eight_Inc.) for providing such great dataset.
* Credits as well to the parent organization: [Appen](https://appen.com/)
