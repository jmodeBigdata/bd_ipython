# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.fftpack as fft
from matplotlib import rc
from math import pi as pi
import sklearn.linear_model as sklm
import time
import sklearn.cross_validation as skcv


def read_csv():
    #sales = pd.read_csv()
    #sales_column_list = ["date", "quantity", "ps", "shoprank", "category", "color", "season", "cost", "brand"]
    #sales.columns = sales_columns_list
    shoprank = pd.DataFrame(pd.get_dummies(sales["shoprank"]))
    category = pd.DataFrame(pd.get_dummies(sales["category"]))
    color = pd.DataFrame(pd.get_dummies(sales["color"]))
    season = pd.DataFrame(pd.get_dummies(sales["season"]))
    brands = pd.DataFrame(pd.get_dummies(slaes["brand"]))

    weather = pd.read_csv("./weather.csv", header=0, encoding="shift-jis")
    weather.columns = ["date","ave_degree","all_rainfall","day_time","1day_before","2day_before","night_weather","day_weather"]
    night_weather = pd.DataFrame(pd.get_dummies(weather["night_weather"]))
    day_weather = pd.DataFrame(pd.get_dummies(weather["day_weather"]))

    weather = pd.concat([weather["date"],weather["ave_degree"],weather["all_rainfall"],weather["day_time"],night_weather,day_weather],axis=1)
    print(weather)

    return sales, weather

def make_Train():
    #商品IDとAREA
    x=234

def make_Test():
    d=1


def Analyze():




if __name__ == '__main__':
    All = read_csv()
    make_Train()
    make_Test()
