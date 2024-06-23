
# Analyzing Factors Affecting RateMyProfessors.com Ratings
###### Bianca Stiles, Maia Han, and Rachel Ferina

### Required Python Packages
These packages must be installed to run this analysis. The specific required imports are listed as well.
- pandas
 - `import pandas as pd`
- geopandas
 - `import geopandas as gpd`
- numpy
 - `import numpy as np`
- matplotlib
 - `import matplotlib.pyplot as plt`
- sklearn
 - `from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor`
 - `from sklearn.tree import export_graphviz`
 - `from sklearn.model_selection import train_test_split`
 - `from sklearn.metrics import mean_squared_error, accuracy_score`
- xgboost
 - `import xgboost`
- seaborn
 - `import seaborn as sns`
- graphviz
 - `import graphviz`
- IPython
 - `from IPython.display import Image, display`
- scipy
 - `import scipy.stats as stats`

### File Descriptions
This analysis contains three files.
- **CSE_163_Ratemyprofessor.py** contains all the functions to process the
RateMyProfessors data for analysis and create machine learning models and graphs.
- **CSE_163_executive_file.py** contains the main function which calls the
functions from **CSE_163_Ratemyprofessor.py**.
 - Running **CSE_163_executive_file.py** executes itself and **CSE_163_Ratemyprofessor.py**.
- **CSE_163_test_file.py** contains the test functions to ensure the functions
in **CSE_163_Ratemyprofessor.py** are working correctly.

### RateMyProfessors Website Background
[ratemyprofessors.com](https://www.ratemyprofessors.com/) is a website where
students can give their professors a rating based on their personal experience
with their classes. It's helpful for prospective students to decide which professor
to take, and also for professors to reflect on their teaching strategies.

### RateMyProfessors Dataset
This Dataset is from the [Mendeley Data website](https://data.mendeley.com/datasets/fvtfjyvw7d/2). The author of the dataset is Jibo He, a professor at Tsinghua University. This data was web crawled from the RateMyProfessors.com website in 2018. The collected data includes ratings from 2003 to 2018, however, the RateMyProfessor.com website was updated in 2015, so there is a lot of missing data for the years prior. Our analysis filters the data to only include the rating posts with dates of 2015 and later as a result. The data only includes currently teaching professors, as RateMyProfessors.com deletes professor profiles once they stop teaching.
- The `school_name` column contains the university at which the professor is currently teaching
- The `state_name` column contains a two character state abbreviation.
- The `year_since_first_review` column contains the number of years the professor has been teaching, from the first post date to 2019 when this calculation was performed.
- The `num_students` column contains the number of students in the course.
- The `post_date` column contains the date when the student posted the rating.
- The `student_star` column contains the rating students gave the professor, out of 5 stars. A rating of 1 is considered to be an awful professor, a rating of 5 is considered to be an awesome professor.
- The `student_difficult` column contains the rating out of 5 for how difficult students considered the professor to be for that course. A rating of 1 means very easy, and a rating of 5 means very difficult.
- The `attence` column contains if attendance for the course was mandatory or not mandatory.
- The `for_credit` column contains if the student took the course for credit (yes or no).
- The `would_take_agains` column contains if students would take that course again (yes or no).
- The `grades` column contains the students final grade in the course as A+, A, A-, B+, B, B-, C+, C, C-, D+, D, D-, F, WD, INC, Not, Audit/No. WD is Drop/Withdrawal. INC means Incomplete. Not is Not sure yet, and Audit/No is Audit/No Grade
- The `IsCourseOnline` column contains if the course was online or not (1 or 0).
- The tags are three adjectives a student can use to describe a professor.
There are several tag columns, and selected tags are as represented as 1. The
tag columns are `gives_good_feedback`, `caring`, `respected`, `participation_matters`,
`clear_grading_criteria`, `skip_class`, `amazing_lectures`, `inspirational`,
`tough_grader`, `hilarious`, `get_ready_to_read`, `lots_of_homework`,
`accessible_outside_class`, `lecture_heavy`, `extra_credit`, `graded_by_few_things`,
`group_projects`, `test_heavy`, `so_many_papers`, and `beware_of_pop_quizzes`

### Country Dataset
This shapefile dataset is from the [National Weather Service website](https://www.weather.gov/gis/USStates), and was derived from U.S. Counties. We included
this dataset in our analysis to merge with the RateMyProfessors dataset, so we
could graph a map using the geometry data. We will use the state column to merge
our datasets, and the geometry to map the merged data.
- The `STATE` column contains a two character state abbreviation.
- The `NAME` column contains the state names.
- The `FIPS` column contains the state FIPS code.
- The `LON` column contains the longitude of the centroid in degrees as a float.
- The `LAT` column contains the latitude of centroid in degrees as a float.

### Test Dataset
This is a smaller subset of the RateMyProfessors dataset that was randomly selected. It is used to test how the functions in **CSE_163_Ratemyprofessor.py** run.
