'''
CSE 163 Final Project
Bianca Stiles
Maia Han
Rachel Ferina
This file analysis all data we have from ratemyprofessors.com
and run through all the functions we imported from the code file
including filtering out the data before year 2015
adding teaching experience column to the data
building two Machine Learnig Models
running statistics computation
generating statistics figures
mapping one of the statistic figures
'''
import CSE_163_Ratemyprofessor as rmp
import warnings


datafile = 'RateMyProfessor_Sample data.csv'
countrydata = 'geodata'


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    dataframe = rmp.openfile(datafile)
    country = rmp.open_geo_file(countrydata)
    # graph school types vs rating
    school_type_ratings = rmp.school_types(dataframe)
    print(rmp.school_type_statistics(school_type_ratings))
    data = rmp.filter_data(dataframe)
    # adding yoe column
    data = rmp.teaching_experience(data)
    # compare sklearn/xgb rating models
    print(rmp.sklearn_rating_model(data))
    print(rmp.xgb_rating_model(data))
    # compare sklearn/xgb take-again medles
    print(rmp.sklearn_take_again_model(data))
    print(rmp.xgb_take_again_model(data))
    # graph rating over year
    rmp.rating_over_year(data)
    rmp.rating_by_states(dataframe, country)
    rmp.mapping(dataframe, country)


if __name__ == "__main__":
    main()
