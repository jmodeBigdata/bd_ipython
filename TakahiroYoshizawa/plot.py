# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt

def read_forecasts():
    FFT1 = pd.read_csv("~/PycharmProjects/MATSUO/forecast_cv110.csv",index_col=0, header=0, names=codeList)
    FFT2 = pd.read_csv("~/PycharmProjects/MATSUO/forecast_cv210.csv",index_col=0, header=0, names=codeList)
    FFT3 = pd.read_csv("~/PycharmProjects/MATSUO/forecast_cv310.csv",index_col=0, header=0, names=codeList)
    FFT4 = pd.read_csv("~/PycharmProjects/MATSUO/forecast_cv410.csv",index_col=0, header=0, names=codeList)
    FFT5 = pd.read_csv("~/PycharmProjects/MATSUO/forecast_cv510.csv",index_col=0, header=0, names=codeList)
    FFT6 = pd.read_csv("~/PycharmProjects/MATSUO/forecast_cv610.csv",index_col=0, header=0, names=codeList)

    ARIMA = pd.read_csv("~/PycharmProjects/MATSUO/ARIMA.csv", index_col=0, header=0, names=codeList)
    ACTUAL = pd.read_csv("~/PycharmProjects/MATSUO/test_3_19.csv", index_col=0, header=0, names=codeList)

    return FFT1, FFT2, FFT3, FFT4, FFT5, FFT6, ARIMA, ACTUAL




if __name__ == '__main__':
    codeList = [1101,1102,1103, 1104, 1105, 1107, 1108, 1109, 1110, 1111, 1220, 1221, 1222, 1223, 1224, 1226, 1227, 1228, 5330, 5331, 5332]
    FFT1f, FFT2f, FFT3f, FFT4f, FFT5f, FFT6f, ARIMAf, ACTUALf = read_forecasts()

    MAPE = pd.DataFrame(None)

    i = 1

    for item in codeList:
        FFT1 = FFT1f[item]
        FFT2 = FFT2f[item]
        FFT3 = FFT3f[item]
        FFT4 = FFT4f[item]
        FFT5 = FFT5f[item]
        FFT6 = FFT6f[item]
        SARIMA = ARIMAf[item]
        ACTUAL = ACTUALf[item]

        #FourierTransform_MAPE = sum(abs(FFT - ACTUAL)/ACTUAL) / 51 * 100
        #SARIMA_MAPE = sum(abs(SARIMA - ACTUAL)/ACTUAL) / 51 * 100
        #FourierTransform_RMSE = (sum((FFT-ACTUAL)**2)/51)**(1/2)
        #SARIMA_RMSE = ((sum((SARIMA -ACTUAL)**2))/51)**(1/2)
        #MAPE[i] = [item, FourierTransform_MAPE,SARIMA_MAPE,FourierTransform_RMSE,SARIMA_RMSE]


        plt.figure()
        FFT1.plot(label="1y Train FT", color="#3D6CCC", lw=2.5)
        FFT2.plot(label="2y Train FT", color="#9C3DCC", lw=2.5)
        FFT3.plot(label="3y Train FT", color="#CC3D6C", lw=2.5)
        FFT4.plot(label="4y Train FT", color="#CC9C3D", lw=2.5)
        FFT5.plot(label="5y Train FT", color="#6CCC3D", lw=2.5)
        FFT6.plot(label="6y Train FT", color="#3DCC9C", lw=2.5)
        ACTUAL.plot(label="Actual", color="#252535", lw=3)

        plt.legend()
        plt.xlabel("Week")
        plt.ylabel("Quantity")
        plt.show()

        i += 1

    #MAPE.to_csv("MAPE_RMSE_3_19.csv",header=False)

"""
        fig = plt.figure()
        ax1=fig.add_subplot(6,1,1)
        FFT1.plot(label="1y Train FT", color="#408000", style="-*",ax=ax1)
        ACTUAL.plot(label="Actual", color="Red")
        ax2 = fig.add_subplot(6,1,2)
        FFT2.plot(label="2y Train FT", color="#FF8000", style="--p",ax=ax2)
        ACTUAL.plot(label="Actual", color="Red")
        ax3 = fig.add_subplot(6,1,3)
        FFT3.plot(label="3y Train FT", color="#0080FF", style="-o",ax=ax3)
        ACTUAL.plot(label="Actual", color="Red")
        ax4 = fig.add_subplot(6,1,4)
        FFT4.plot(label="4y Train FT", color="#004080", style="-s",ax=ax4)
        ACTUAL.plot(label="Actual", color="Red")
        ax5 = fig.add_subplot(6,1,5)
        FFT5.plot(label="5y Train FT", color="#400080", style="--s",ax=ax5)
        ACTUAL.plot(label="Actual", color="Red")
        ax6 = fig.add_subplot(6,1,6)
        FFT6.plot(label="Fourier Transform", color="b", style="-^",ax=ax6)
        #SARIMA.plot(label="SARIMA", color="Green")
        ACTUAL.plot(label="Actual", color="Red")
"""
