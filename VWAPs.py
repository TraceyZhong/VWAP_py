# 一致用系统的时间！经过思考这个是最吼的！

"""
This module provide a VWAPs object tracking multiple tickers trading volume and 
predict trading percentage the next time interval.

We first predict the total trading volume, then dynamically predict each interval's 
trading volume by an AR1 model. The trading percentage is first predicted by
predicted volume divided by the predicted total trading volume, but further adjusted
by the expected finished trading precentage at that specific time.

You must provide at least 15 days of tick data (with volumes traded in call auction.)

You should provide an full path to directory of all tick data whose subdirectories 
contain that for each stock. Typically, each subdirectory should be named by its trade ID
with letters in upper case. Each file should start with its trade ID followed by date.

Example: full_path/SH000001/SH00000012018-01-10.csv. CSV format is required.

Each file should only contain two columes splitted by comma: the first the traded time
the second volumes.

Example: 10:23:12, 1234
"""


import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta

import scipy.interpolate
from sklearn.linear_model import Lasso
from statsmodels.tsa.arima_model import ARMA

import warnings
from os import listdir


def cov(a,b):

    a_mean = np.mean(a)
    b_mean = np.mean(b)
    sum = 0
    for i in range(0,len(a)):
        sum += (a[i] - a_mean)*(b[i] - b_mean)
    
    return sum / (len(a) - 1)


def getL(y):
    """By linear regression predict the next value"""

    x = np.array(range(0,len(y)))
    b = cov(x,y) / cov(x,x)
    a = np.mean(y) - b * np.mean(x)
    
    return b * len(y) + a


def rolling_mean(a,n = 5):

    x = [0.] * (len(a)-n + 1)
    for i in range(n,len(a) +  1):
        x[i - n] = a[i-n:i].mean()
    
    return x


def rolling_linear(a, n = 5):
    
    x = [0.] * (len(a)-n + 1)
    for i in range(n , len(a)+ 1):
        x[i - n] = getL(a[i-n:i])
    
    return x


def get_log(r_vol,p_vol,p_per): 
    
    return {'r_vol':r_vol, 'p_vol':p_vol, 'p_per':p_per}


def datetime_range(T_START_TIME, T_END_TIME ,delta):
    
    current = T_START_TIME
    end = T_END_TIME
    while current < end:
        yield current
        current += delta


class VWAP(object):
    """
    a single VWAP object will track and predict one ticker

    Args:
        HALFTIME (time):  half of the trading hours
    """


    HALFTIME = timedelta(hours = 2)

    def __init__(self, interval, ticker, kwargs):
        '''
        Collect historical data and calculate parameters for volume predictions and AR1

        Args:
            TODAY: a datetime object of today
            T_START_TIME: today's market opening time
            T_END_TIME: today's market closing time
            LASSO_LAMBDA: lambda of lasso method
            N_TICK_THRESHOULD: the least number of tick data a valid file should contain
            DATA_PATH: the absolute path to tick data for this ticker
            _interval: how many secs to update volume percentage prediction
            _interval_timedelta: interval in timedelta format
            _semi_n_interval: half of the total number of intervals today
            _n_interval: the total number of intervals today 
            _features_to_train: 10 valid days of CA, rolling mean and rolling linear prediction
            _histo_volume: 10 valid days of traded volume in each interval
            _intraday_percentage: expected trading volumn in each interval
            _AR_pars_pm: mu and phi for AR(1) in the morning
            _AR_pars_pm: mu and phi for AR(1) in the afternoon
            _CA_today: todays traded vulumes during todays call auction
            _predicted_V: today's total predicted trading volume
            _is_V_predicted: a flag of is V predicted
            _iter: in which interval the current pushed tick
            _last_update: the interval last updated in VWAP_log
            _datetime_index: the index for VWAP_log
            _today_vol: a list of today's trading volume in each interval
            _p_per: a list of predicted trading volume percentage
            _p_vol: a list of predicted trading volume
            _VWAP_log: the log file of the predicted volume volume and its percentage and true volume in each interval
        
        Methods:
            pred_V: predict today's total trading volume
            push_tick: push tick data
            get_prediction: print out  
        '''
        if (interval % 5 != 0) or (7200 % interval != 0):
            raise ValueError('interval must be a multiple of 5 secs and can divide 2 hours')

        if not ticker in listdir(kwargs['DATA_PATH']):
            raise Exception('no data for %s' % ticker)
        
        self.TODAY = kwargs['TODAY']
        # self.TODAY = datetime.strptime(today_for_test, "%Y-%m-%d")  # Tracey to notice
        self.T_START_TIME = kwargs['T_START_TIME']
        # self.T_START_TIME = self.TODAY.replace(hour = 9, minute = 30, second = 0, microsecond = 0)
        self.T_END_TIME = kwargs['T_END_TIME']
        # self.T_END_TIME = self.TODAY.replace(hour = 15, minute = 00, second = 0, microsecond = 0)
        self.LASSO_LAMBDA = kwargs['LASSO_LAMBDA']
        self.N_TICK_THRESHOLD = kwargs['N_TICK_THRESHOLD']
        self.DATA_PATH = kwargs['DATA_PATH'] + ticker + '/'
        # self.DATA_PATH = './data_path/' # Tracey to notice
        self._interval = interval
        self._interval_timedelta = timedelta(seconds = self._interval)
        self._semi_n_interval = int(self.HALFTIME / self._interval_timedelta)
        self._n_interval = 2 * self._semi_n_interval
        self._features_to_train = np.ones((11,3),dtype=float) # CA, M, L, A
        self._histo_volume = np.full((10,self._n_interval),0, dtype=float)  # historical trading volume        
        self._intraday_percentage = [1 / self._n_interval] * self._n_interval  # notice .sum() =self._n_interval
        # self._AR_pars = np.array([1,0],dtype =float) # (u and phi)
        self._AR_pars = [0., 1.] 
        self._CA_today = 0
        self._predicted_V = 0.
        self._is_V_predicted = 0
        self._last_update = 0
        self._iter = 0       
        self._datetime_index = ( [str(dt) for dt in datetime_range(self.T_START_TIME, 
                                self.T_START_TIME.replace(hour = 11, minute = 30, second = 0, 
                                microsecond = 0),timedelta(seconds = self._interval))] + 
                                [str(dt) for dt in datetime_range(self.T_START_TIME.replace(hour = 13, 
                                minute = 0, second = 0, microsecond = 0), 
                                self.T_END_TIME,timedelta(seconds = self._interval))])
        self._today_vol = [0.] * self._n_interval
        self._p_per = [0.] * self._n_interval
        self._p_vol = [0] * self._n_interval
        self._VWAP_log = {}
        
        files = set([ filename for filename in listdir(self.DATA_PATH) if filename.endswith( '.csv' ) ])
        history_date = self.TODAY
        x_output = np.append(np.arange(0 + self._interval , 7200 + self._interval, self._interval), 
                            np.arange(12600 + self._interval,19800 + self._interval,self._interval))
        
        past_days = 0
        iter = 1
        
        # get data for intraday prediction
        while iter < 11:

            if not bool(files):
                raise Exception('Insufficient historical data')
            
            history_date = history_date - timedelta(days = 1)
            past_days += 1
            
            if history_date.weekday() in set([5,6]):
                continue
            
            filename = str(ticker) + str(history_date.strftime('%Y-%m-%d'))+'.csv' ## Tracey to notice
            
            if filename in files:
                files.remove(filename)
            else:
                continue

            try:
                dat = pd.read_csv(self.DATA_PATH+filename)
            except Exception:
                print('Error in reading %s, go to the previous day.' % filename)
                continue

            if dat.shape[0] < self.N_TICK_THRESHOLD:
                print('File %s has few data for prediction' % filename)
                continue

            if past_days > 20:
                warnings.warn('Lack historical data. Time span of data for predicting intraday_volume of today has exceeded 20 days.'
                                'We are using data %d days from today' % past_days)        
            
            try:
                dat.columns = ['DateTime','Volume'] # there will be Microsecond
                dat.DateTime = [datetime.strptime(str(history_date.strftime('%Y-%m-%d')) + ' ' + dt, "%Y-%m-%d %H:%M:%S") for dt in dat.DateTime]
                
                # datetime to time difference
                self.H_START_TIME = history_date.replace(hour = 9, minute = 30, second = 0, microsecond = 0)            
                dat['TimeStamp'] = [(dt - self.H_START_TIME).total_seconds() for dt in dat.DateTime]
                dat = dat.as_matrix(columns = ['TimeStamp','Volume'])
                
                datCA = dat[dat[:,0] < 0]
                self._features_to_train[10 - iter,0] = datCA[:,1].sum()
                dat = dat[dat[:,0] > 0]

                # Tracey by reviewing the data from ctp finds it impossible
                if any(t >= 198000 for t in dat[:,0]):
                    dat = np.vstack((dat[dat[:,0]<19800],[19800,dat[dat[:,0] >= 19800,1].sum()]))
                dat[-1,0] = 198000
                x_input = np.append(0, dat[:,0])
                volume_cumsum = np.append(0,dat[:,1].cumsum())
                y_interp = scipy.interpolate.interp1d(x_input,volume_cumsum) # ,interval)
                intraday_volume = y_interp(x_output)
                intraday_volume = np.append(intraday_volume[0],(intraday_volume[1:] - intraday_volume[:-1]))
                self._histo_volume[10 - iter] = intraday_volume
            except Exception:
                print('Error when read file %s, you may check its format' % filename)
                continue

            iter += 1

        iter = 11 # 这个不需要

        # get data for roll_mean and roll_linear
        volume_sums = np.zeros(5,dtype=float)
        history_date = self.TODAY -  timedelta(days = past_days)
        while iter < 16:

            if not bool(files):
                raise Exception('Insufficient historical data')
            
            history_date = history_date - timedelta(days = 1)
            past_days += 1
            
            if history_date.weekday() in set([5,6]):
                continue
            
            # filename = str(history_date.strftime('%Y-%m-%d'))+'.csv' ## Tracey to notice
            filename = str(ticker) + str(history_date.strftime('%Y-%m-%d'))+'.csv' ## Tracey to notice
            if filename in files:
                files.remove(filename)
            else:
                continue

            try:
                dat = pd.read_csv(self.DATA_PATH+filename)
            except Exception:
                print('Error in reading %s, go to the previous day.' % filename)
                continue

            if past_days > 30:
                warnings.warn('Lack historical data. Time span of data for predicting total trading volume of today has exceeded 30 days.')        
            
            try:
                dat = pd.read_csv(self.DATA_PATH+filename)
                dat.columns = ['DateTime','Volume']
                self.H_START_TIME = history_date.replace(hour = 9, minute = 30, second = 0, microsecond = 0)
                dat.DateTime = [datetime.strptime(str(history_date.strftime('%Y-%m-%d')) + ' ' + dt, 
                                                        "%Y-%m-%d %H:%M:%S") for dt in dat.DateTime]
                volume_sums[15 - iter] = dat[dat.DateTime > self.H_START_TIME].Volume.sum()
            except Exception:
                print('Error when read file '+ filename + ', you may check its format')
                continue
            
            iter += 1

        # preparing sample for predicting today's total volume
        self.volume_to_train = self._histo_volume.sum(axis = 1)
        volume_sums = np.append(volume_sums, self.volume_to_train)
        self._features_to_train[:,1] = rolling_mean(volume_sums)
        self._features_to_train[:,2] = rolling_linear(volume_sums)

        # get intraday pattern and intialize intraday prediction
        intraday_mean = self._histo_volume.mean(axis = 0)
        self._p_vol[0] = float(intraday_mean[0])
        self._p_vol[self._semi_n_interval] = float(intraday_mean[self._semi_n_interval])
        self._intraday_percentage = list(np.divide(intraday_mean, intraday_mean.sum()) * self._n_interval)
        if any( i < 1 / (self._n_interval * 10 ) for i in self._intraday_percentage):
            warnings.warn('adjust intraday trading volume pattern for irregular data')
        tmp = np.divide(intraday_mean, intraday_mean.sum()) * self._n_interval
        if np.any(tmp < 0.1):
            warnings.warn('adjust intraday trading volume pattern for irregular data')
            tmp[tmp >= 0.1] = tmp[tmp > 0.1] * sum(self._n_interval - tmp[tmp < 0.1]) / sum(tmp[tmp >= 0.1])
            tmp[tmp < 0.1] = 0.1
        self._intraday_percentage = list(tmp)
        self._p_per[0] = self._intraday_percentage[0] / self._n_interval
        self._p_per[self._semi_n_interval] = self._intraday_percentage[self._semi_n_interval] / self._n_interval        
        self._VWAP_log[self._datetime_index[0]] = get_log(None, self._p_vol[0], self._p_per[0])         
        
        # compute AR
        arma = ARMA( (self._histo_volume[-1] / self._intraday_percentage)[0:self._n_interval], order = (1,0))
        self._AR_pars = arma.fit().params.tolist()

    def pred_V(self):
        
        if self._CA_today == 0:
            self._features_to_train[10,0] = self._features_to_train[:,0].sum()
        else:
            self._features_to_train[10,0] = self._CA_today
        
        lm = Lasso(alpha = self.LASSO_LAMBDA)
        lm.fit(self._features_to_train[0:-1,:],self.volume_to_train)
        self._predicted_V = int(lm.predict(self._features_to_train[-1].reshape(1,-1))[0])
        if self._predicted_V < 0:
            warnings.warn('We some how get a exceeding low volume prediction for today. We strongly urge you check your tick data.')
            self._predicted_V = 1 # Tracey to notice
        self._is_V_predicted = 1
        print('finish: pred_V')

    def push_tick(self, date_time, volume):
        
        if date_time < self.T_START_TIME:
            self.CAtoday += volume
        
        elif date_time < self.T_END_TIME: 
            
            if not self.is_V_predicted:
                self.pred_V()
            
            iter = int((date_time - self.T_START_TIME) / self.INTERVAL)
            
            if iter > (self._semi_n_interval): # in the afternoon
                iter -= int(self._n_interval  * 3 / 8)
            self._today_vol[iter] += volume
            self._iter = iter
            
            if self._iter == self._last_update:
                pass
            
            elif self._iter - self._last_update == 1:
                self.VWAP_log[self._datetime_index[self._last_update]] = get_log(self._today_vol[self._last_update], self._p_vol[self._last_update], self._p_per[self._last_update])
                self._p_vol[self._iter] = int ((self._AR_pars[1] * (self._today_vol[self._last_update] / self.intraday_percentage[self._last_update] - self._AR_pars[0] ) + self._AR_pars[0] ) * self.intraday_percentage[self._iter])
                
                if self._iter < (self._n_interval - 1):
                    self._p_per[self._iter] = self._p_vol[self._iter] * (1 - sum(self._p_per[0:self._iter])) / (self._predicted_V * (1 - sum(self.intraday_percentage[0:self._iter])/ self._n_interval ))
                else:
                    self._p_per[self._n_interval - 1] = 1 - sum(self._p_per[0:(self._n_interval - 1)])
                
                self.VWAP_log[self._datetime_index[self._iter]] = get_log(None, self._p_vol[self._iter], self._p_per[self._iter])
                self._last_update = self._iter
            
            elif self._iter - self._last_update > 1:
                warnings.warn('Over %d secs without receiving data' % self.interval)
                self._today_vol[iter] =+ volume
                self._today_vol[self._last_update:self._iter] = [a + b for a, b in zip(self._today_vol[self._last_update:self._iter], [volume * s / sum(self.intraday_percentage[self._last_update:self._iter]) for s in self.intraday_percentage[self._last_update:self._iter]])]
                
                for i in range(self._last_update, self._iter):
                    self.VWAP_log[self._datetime_index[i]] = get_log(self._today_vol[i], self._p_vol[i], self._p_per[i])
                    self._p_vol[i + 1] = int ((self._AR_pars[1] * (self._today_vol[i] / self.intraday_percentage[i] - self._AR_pars[0] ) + self._AR_pars[0] ) * self.intraday_percentage[i + 1]) 
                    if i + 1 < (self._n_interval - 1):
                        self._p_per[i + 1] = self._p_vol[i + 1] * (1 - sum(self._p_per[0:(i + 1)])) / (self._predicted_V * (1 - sum(self.intraday_percentage[0:(i + 1)])/ self._n_interval ))
                    else:
                        self._p_per[self._n_interval - 1] = 1 - sum(self._p_per[0:(self._n_interval - 1)])
                    self.VWAP_log[self._datetime_index[i + 1]] = get_log(None, self._p_vol[i + 1], self._p_per[i + 1])
                self._last_update = self._iter                    
            
            else: # when self._iter < self._last_update, we only update real volume
                pass
        
        else: # datetime > T_END_TIME
            if self._iter - self._last_update == 1:
                pass 
            else:
                warnings.warn('Over %d secs without receiving data' % self.interval)
                self._today_vol[iter] =+ volume
                self._today_vol[self._last_update:self._iter] = [a + b for a, b in zip(self._today_vol[self._last_update:self._iter], [volume * s / sum(self.intraday_percentage[self._last_update:self._iter]) for s in self.intraday_percentage[self._last_update:self._iter]])]
                
                for i in range(self._last_update, self._n_interval - 1):
                    self.VWAP_log[self._datetime_index[i]] = get_log(self._today_vol[i], self._p_vol[i], self._p_per[i])
                    self._p_vol[i + 1] = int ((self._AR_pars[1] * (self._today_vol[i] / self.intraday_percentage[i] - self._AR_pars[0] ) + self._AR_pars[0] ) * self.intraday_percentage[i + 1]) 
                    
                    if i + 1 < (self._n_interval - 1):
                        self._p_per[i + 1] = self._p_vol[i + 1] * (1 - sum(self._p_per[0:(i + 1)])) / (self._predicted_V * (1 - sum(self.intraday_percentage[0:(i + 1)])/ self._n_interval ))
                    else:
                        self._p_per[self._n_interval - 1] = 1 - sum(self._p_per[0:(self._n_interval - 1)])
                    
                    self.VWAP_log[self._datetime_index[i + 1]] = get_log(None, self._p_vol[i + 1], self._p_per[i + 1])
                
                self._last_update = self._n_interval - 1 
                
            self._today_vol[self._n_interval - 1] += volume 
            self.VWAP_log[self._datetime_index[self._n_interval - 1]] = get_log(self._today_vol[self._n_interval - 1], self._p_vol[self._n_interval - 1], self._p_per[self._n_interval - 1])            


    def get_predict(self):
        return(self._VWAP_log)


class VWAPs(object):

    """
    Args:
        interval (int): how many secs to update volume percentage prediction
        tickers (list of str): tickers whose letters are upper cased
        data_path (str): absolute data_path for all tick data and end with a slash
        lasso_lambda (float): the parameter to predict today's trading volume, default = 812314
        n_tick_threshold (int) : the least number of tick data a valid file should contain, default = 1000

    Examples:
    Tracey = VWAPs(30, ['SH00019','SH000018'], lasso_lambda = 812314)
    Tracey['SH00019'].push_tick(10:10:10, 123)
    Tracey['SH00018'].get_prediction()
    """

    def __init__(self, interval, tickers, data_path ,lasso_lambda = 812314, n_tick_threshold = 1000):

        self.tickers = {}
        self._params = {
            'TODAY': datetime.today(),
            'T_START_TIME': datetime.today().replace(hour = 9, minute = 30, second = 0, microsecond = 0),
            'T_END_TIME': datetime.today().replace(hour = 15, minute = 0, second = 0, microsecond = 0),
            'LASSO_LAMBDA': lasso_lambda,
            'N_TICK_THRESHOLD': n_tick_threshold,
            'DATA_PATH': data_path  
        }
        for ticker in tickers:
            self.tickers[ticker] = VWAP(interval, ticker, self._params)
