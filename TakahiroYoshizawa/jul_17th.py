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

    climate = pd.read_csv("./weather.csv", header=0, encoding="shift-jis")



    print(climate)


    return climate

def make_Train():
    #商品IDとAREA
    x=234


if __name__ == '__main__':
    All = read_csv()
