'''
CSE 163 Final Project
Bianca Stiles
Maia Han
Rachel Ferina
This file contains all the functions for analysing data of ratemyprofessors.com
including filtering out the data before year 2015
adding teaching experience column to the data
building two Machine Learnig Models
running statistics computation
generating statistics figures
mapping one of the statistic figures
'''
import scipy.stats as stats
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
import xgboost
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import geopandas as gpd
from IPython.display import Image, display
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

sns.set()


datafile = 'RateMyProfessor_Sample data.csv'
countrydata = 'geodata'


# open two data files
def openfile(filename):
    '''
    takes a csv filename and read the data into a dataframe
    '''
    data = pd.read_csv(filename)
    return data


def open_geo_file(filename):
    country = gpd.read_file(filename)
    return country


# filter our original csv data file to keep only data after 2015 (inclusive)
def filter_data(data):
    '''
    takes in our original dataframe and returns the dataframe we want to use
    for building ML models
    this function extracts comments posted >= 2015
    add a new column "post_year" with intergers showing the post
    year of the comment
    '''
    mask = data['post_date'].notna()
    data = data[mask]
    data_dic = data.to_dict("records")
    # change dataframe to a list of dictionaries
    for row in data_dic:
        # only keep the later four digits in post date to get year only
        if str(row['post_date'])[-4:] >= str(2015):
            row['post_year'] = str(row['post_date'])[-4:]
        else:
            row['post_year'] = 'before 2015'
    # change the list of dictionaries back to dataframe
    df = pd.DataFrame(data_dic)
    df = df[df['post_year'] != 'before 2015']
    # change type of year from str to int
    df['post_year'] = df['post_year'].astype('int')
    return df


# add teaching experience column
def teaching_experience(data):
    '''
    recalculate the year of teaching experience for
    each comment based on the post year return the dataframe with
    a new column "yoe" showing teaching experience
    '''
    data['yoe'] = data['year_since_first_review'] - (2019 - data['post_year'])
    data = data[data['yoe'] > 0]
    return data


# build two models in sklearn and xgboost then compare the MSE of models
# for predicting rating
def sklearn_rating_model(data):
    '''
    Train a DecisionTreeRegressor to predict the rating of a professor
    based on the information in the comment 'would_take_agains'
    '''
    data = data[['yoe', 'num_student',
                 'student_star', 'student_difficult', 'attence',
                 'for_credits', 'grades',
                 'gives_good_feedback', 'caring', 'respected',
                 'participation_matters', 'clear_grading_criteria',
                 'skip_class', 'amazing_lectures', 'inspirational',
                 'tough_grader', 'hilarious', 'get_ready_to_read',
                 'lots_of_homework', 'accessible_outside_class',
                 'lecture_heavy', 'extra_credit', 'graded_by_few_things',
                 'group_projects', 'test_heavy', 'so_many_papers',
                 'beware_of_pop_quizzes', 'IsCourseOnline']]
    data = data.dropna()
    # define features
    X = data.loc[:, data.columns != 'student_star']
    # define labels
    y = data['student_star']
    X = pd.get_dummies(X)
    # split the data into training and testing groups (70:30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    # Create an untrained model
    model = DecisionTreeRegressor()
    # Train it on our training data
    model.fit(X_train, y_train)
    # make predictions on test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    return ('skl Train MSE:', mse_train
            ), ('skl Test MSE:', mse_test)


def xgb_rating_model(data):
    '''
    use xgboost to build a ML model, get MSE for training and test,
    also plots a feature importance ranking figure and our
    decision tree figure
    '''
    data = data[['yoe', 'num_student',
                 'student_star', 'student_difficult', 'attence',
                 'for_credits', 'grades',
                 'gives_good_feedback', 'caring', 'respected',
                 'participation_matters', 'clear_grading_criteria',
                 'skip_class', 'amazing_lectures', 'inspirational',
                 'tough_grader', 'hilarious', 'get_ready_to_read',
                 'lots_of_homework', 'accessible_outside_class',
                 'lecture_heavy', 'extra_credit', 'graded_by_few_things',
                 'group_projects', 'test_heavy', 'so_many_papers',
                 'beware_of_pop_quizzes', 'IsCourseOnline']]
    data = data.dropna()
    # define features
    features = data.loc[:, data.columns != 'student_star']
    # define labels
    label = data['student_star']
    features = pd.get_dummies(features)
    # get training group and testing group (70:30)
    features_train, features_test, label_train, label_test = \
        train_test_split(features, label, test_size=0.30)
    # Create an untrained model
    model = xgboost.XGBRegressor()
    # Train it on our training data
    model.fit(features_train, label_train)
    # build a short model for ploting decision tree figure
    short_model = DecisionTreeRegressor(max_depth=4)
    # Train it on the training set
    short_model.fit(features_train, label_train)
    plot_tree(short_model, features_train, label_train,
              'rating_decision_tree_4')
    # plot the feature importance graph
    importances = model.feature_importances_
    indices = np.argsort(importances)
    # change the figure size accordingly
    fig, ax = plt.subplots(1, figsize=(26, 13))
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(features_train.columns)[indices])
    plt.xlabel("Importance Percentage", fontsize=20)
    plt.ylabel("Features", fontsize=20)
    plt.title("Ranking of Most Important Features in Rating Model",
              fontsize=25)
    plt.savefig('feature importance of rating model.png')
    # make predictions on test data
    y_train_pred = model.predict(features_train)
    y_test_pred = model.predict(features_test)
    # Compute training MSE
    mse_train = mean_squared_error(label_train, y_train_pred)
    mse_test = mean_squared_error(label_test, y_test_pred)
    return ('xgb Train MSE:', mse_train
            ), ('xgb Test MSE:', mse_test)


# build two models in sklearn and xgboost then compare the Accuracy of models
# for predicting if a student would take a course again
def sklearn_take_again_model(data):
    # selecting relevant columns for predicting if a student
    # would take the course again
    data = data[['yoe', 'student_difficult', 'num_student', 'attence',
                 'for_credits', 'grades', 'IsCourseOnline',
                 'would_take_agains']]
    data = data.dropna()
    labels = data['would_take_agains']
    features = data.loc[:, data.columns != 'would_take_agains']
    # convert columns to numerical
    features = pd.get_dummies(features)
    # Create an untrained model
    model = DecisionTreeClassifier()
    # split data into 70% training data, 30% test data
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)
    # train the model with the training data
    model.fit(features_train, labels_train)
    # Make predictions on the data
    train_prediction = model.predict(features_train)
    test_prediction = model.predict(features_test)
    # Assess the accuracy of the model with training data
    training_accuracy = accuracy_score(labels_train, train_prediction)
    # Assess the accuracy of the model with test data
    test_accuracy = accuracy_score(labels_test, test_prediction)
    return ('skl training accuracy: ', training_accuracy
            ), ('skl testing accuracy: ', test_accuracy)


def xgb_take_again_model(data):
    '''
    this function makes a model to predict if a student would take
    a course a gain using xgboost
    also plots a feature importance ranking figure and our
    decision tree figure
    '''
    # selecting relevant columns for predicting if a student
    # would take the course again
    data = data[['yoe', 'student_difficult', 'num_student', 'attence',
                 'for_credits', 'grades', 'IsCourseOnline',
                 'would_take_agains']]
    data = data.dropna()
    labels = data['would_take_agains']
    features = data.loc[:, data.columns != 'would_take_agains']
    # convert columns to numerical
    features = pd.get_dummies(features)
    # Create an untrained model
    model = xgboost.XGBClassifier()
    # split data into 70% training data, 30% test data
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)
    # train the model with the training data
    model.fit(features_train, labels_train)
    # make a decision tree figure
    short_model = DecisionTreeClassifier(max_depth=4)
    # Train it on the training set
    short_model.fit(features_train, labels_train)
    plot_tree(short_model, features_train,
              labels_train, "take_again_decision_tree_4")
    # graph feature importance
    # get importance
    importances = model.feature_importances_
    # summarize feature importance
    indices = np.argsort(importances)
    fig, ax = plt.subplots(1, figsize=(26, 13))
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(features_train.columns)[indices])
    plt.xlabel("Importance Percentage", fontsize=20)
    plt.ylabel("Features", fontsize=20)
    plt.title("Ranking of Important Features in Take-again Model", fontsize=25)
    plt.savefig("feature importance of take-again model.png")
    # Make predictions on the data
    train_prediction = model.predict(features_train)
    test_prediction = model.predict(features_test)
    # Assess the accuracy of the model with training data
    training_accuracy = accuracy_score(labels_train, train_prediction)
    # Assess the accuracy of the model with test data
    test_accuracy = accuracy_score(labels_test, test_prediction)
    return ('xgb training accuracy: ', training_accuracy
            ), ('xgb testing accuracy: ', test_accuracy)


def rating_over_year(data):
    '''
    this function takes our filtered data (including NaN for student_star
    column) and returns a figure showing the distribution of rating changing
    over year of teaching experience
    '''
    data = data.groupby("yoe")['student_star'].mean()
    data = data.dropna()
    fig, ax = plt.subplots(1, figsize=(20, 10))
    data.plot(x='yoe', y='student_star', kind='line', ax=ax, marker='o')
    plt.xlabel('Year of Teaching')
    plt.ylabel('Rating')
    plt.title("Rating VS Teaching Experience")
    plt.savefig('experience and rate.png')


def plot_tree(model, features, labels, outputname):
    '''
    this funciton plots decision tree for input ML model
    '''
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=features.columns,
                               class_names=labels.unique(),
                               impurity=False,
                               filled=True, rounded=True,
                               special_characters=True)
    graphviz.Source(dot_data).render(outputname + '.gv', format='png')
    display(Image(filename=outputname + ".gv.png"))


def school_types(data):
    '''
    this function takes data from all years and
    groups comments into different school types,
    make a boxplot graph that compares their average star_ratings
    '''
    data_dic = data.to_dict('records')
    ivy = ['Brown University', 'Harvard University', 'Cornell University',
           'Princeton University', 'Dartmouth College', 'Yale University',
           'Columbia University', 'University of Pennsylvania']
    for row in data_dic:
        if row['school_name'] in ivy:
            row['school_type'] = 'Ivy League'
        elif "College" in row['school_name']:
            row['school_type'] = 'Community College'
        else:
            row['school_type'] = 'Other School'
    data = pd.DataFrame(data_dic)
    school_type_data = data[['school_type', 'star_rating']]
    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.boxplot(x='school_type', y='star_rating', data=data, ax=ax)
    plt.xticks(fontsize=20, rotation=0)
    plt.title('Average Rating by School Types', fontsize=25)
    plt.ylabel('Rating', fontsize=20)
    plt.xlabel(None)
    plt.savefig('Rating Difference by School Types boxplot.png')
    return school_type_data


def school_type_statistics(data):
    '''
    this function will calculate the P-value for the differences between
    the average rating for school types using Kruskal Test.
    '''
    data['index1'] = data.index
    data = pd.pivot_table(data, values='star_rating',
                          columns='school_type',
                          index="index1")
    # stats f_oneway functions takes the groups as input and returns
    # ANOVA F and p value
    data_cc = data.dropna(subset=['Community College'])
    data_ivy = data.dropna(subset=['Ivy League'])
    data_os = data.dropna(subset=['Other School'])
    # try p-value computing with anova test and kruskal test
    # fvalue, pvalue = stats.f_oneway(data_cc['Community College'],
    #                                 data_ivy['Ivy League'],
    #                                 data_os['Other School'])
    # print("f:", fvalue, "p-value:", pvalue)
    # print("cc:", len(data_cc), "ivy:", len(data_ivy), "os:", len(data_os))
    kruskal_result = stats.kruskal(data_cc['Community College'],
                                   data_ivy['Ivy League'],
                                   data_os['Other School'])
    return kruskal_result


def rating_by_states(data, country):
    '''
    this funciton graphs average rating for each state in the U.S.
    '''
    state_name = data[['state_name', 'star_rating']].copy()
    country_copy = country.copy()
    # removes spaces from string
    state_name_obj = state_name.select_dtypes(['object'])
    state_name[state_name_obj.columns] = \
        state_name_obj.apply(lambda x: x.str.strip())
    # merge the state names by comparing it to official country data and drop
    # the unmatched ones are taken off
    merge_data = country_copy.merge(state_name, left_on='STATE',
                                    right_on='state_name', how='inner')
    merge_state = merge_data[['star_rating', 'state_name']]
    merge_state = merge_state.dropna(subset=['state_name'])
    # draw the plot
    ratings = merge_state.groupby('state_name')['star_rating'].mean()
    ratings = ratings.sort_values(ascending=False)
    fig, ax = plt.subplots(1, figsize=(20, 10))
    color_list = ['#87CEEB'] * 5 + ['grey'] * 35 + ['#87CEEB'] * 5
    ratings.plot(x="state_name", y="star_rating", kind="bar", color=color_list)
    plt.xticks(fontsize=8, rotation=90)
    plt.xlabel("State", fontsize=20)
    plt.ylabel("Rating", fontsize=20)
    plt.title("Average Rating by State", fontsize=25)
    plt.savefig('rating by states.png')


def mapping(data, country):
    '''
    this function takes a geodata and our unfiltered data to
    generate a map that shows the average ratings of each
    state in the U.S. with color gradients
    '''
    country_copy = country.copy()
    # filters columns needed from data
    ratings = data[['state_name', 'star_rating']].copy()
    # calculates the average star rating by state
    ratings = ratings.groupby('state_name')['star_rating'].mean()
    # turns series back into dataframe
    new_ratings = pd.DataFrame(ratings)
    # Makes index column into dataframe column called state_name
    new_ratings = new_ratings[1:].reset_index().rename(columns={
                                                       'index': 'state_name'})
    # Removes spaces from state name
    state_name_obj = new_ratings.select_dtypes(['object'])
    new_ratings[state_name_obj.columns] = state_name_obj.apply(lambda x:
                                                               x.str.strip())
    # Merges U.S. map data and Ratings data
    data1 = country_copy.merge(new_ratings, left_on='STATE',
                               right_on='state_name', how='left')
    data2 = data1.copy()
    # change the type of Longtitude and the latitude
    data2['LON'] = data2['LON'].astype(np.int64)
    data2['LAT'] = data2['LAT'].astype(np.int64)
    filter_state = data2[(data2["LAT"] <= 50) & (data2["LAT"] >= 20) &
                         (data2["LON"] <= -40) & (data2["LON"] >= -140)]
    fig, ax = plt.subplots(1, figsize=(20, 10))
    filter_state.plot(ax=ax, color="#EEEEEE")
    filter_state.plot(ax=ax, column="star_rating", legend=True)
    plt.title("Average Rating by States")
    plt.savefig('map.png')
