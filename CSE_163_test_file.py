'''
CSE 163 Final Project
Bianca Stiles
Maia Han
Rachel Ferina
This file runs a small test for analysing data of ratemyprofessors.com
by extracting a small fraction of the original dataset,
and run through all the functions below for the small test dataset
including filtering out the data before year 2015
adding teaching experience column to the data
building two Machine Learnig Models
running statistics computation
generating statistics figures
mapping one of the statistic figures
'''
import CSE_163_Ratemyprofessor as rmp


datafile = 'RateMyProfessor_Sample data.csv'
countrydata = 'geodata'


def get_sample(data):
    '''
    this function gets a small random sample data for our testing purpose
    '''
    sample_data = data.sample(frac=.25)
    return sample_data


def test_filter_data(data):
    '''
    this function tests our filter function, which should
    return a dataframe without data before 2015 and have a
    new column post_year
    compare the length of data before 2015 and 0
    the test is passed if returns True
    '''


def main():
    dataframe = rmp.openfile(datafile)
    # get a small random sample data
    test_dataframe = get_sample(dataframe)
    # test the length of the test_dataframe to see if it's 25% of
    # our original dataframe
    # test passed if the outputs are True
    print(len(test_dataframe) == len(dataframe) * 0.25)
    # run our test file to see if the test is True
    # test passed if the outputs are True
    filtered_data = rmp.filter_data(test_dataframe)
    print(len(filtered_data[filtered_data['post_year'] < 2015]) == 0)
    # add year of experience column to the filtered data
    test_dataframe = rmp.teaching_experience(filtered_data)
    # test ML model accuracy/MSE
    print(rmp.xgb_rating_model(test_dataframe))
    print(rmp.xgb_take_again_model(test_dataframe))


if __name__ == "__main__":
    main()
