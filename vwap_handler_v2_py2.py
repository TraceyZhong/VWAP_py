# -*- coding: utf-8 -*-

# required packages

import numpy as np
import pandas as pd

import time
from datetime import datetime
from datetime import timedelta
from datetime import date as date
from datetime import time as dt_time

import scipy.interpolate
from sklearn.linear_model import Lasso
from statsmodels.tsa.arima_model import ARMA

from math import ceil
import warnings
from os import listdir 

# helper functions

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

    current = datetime.combine(date.today(), T_START_TIME)
    end = datetime.combine(date.today(), T_END_TIME)
    while current < end:
        yield current
        current += delta

class VWAP_handler(object):
    """
    a single VWAP object will track and predict one ticker
    """

    def __init__(self, interval, tickers, data_path ,lasso_lambda = 812314, n_tick_threshold = 1000, market_close_time = '15:00:00', n_hist_day = 15):

        self.tickers = {}
        self._params = {
            'TODAY': date.today(),
            # 'TODAY': datetime.strptime(today_for_test, "%Y%m%d"),
            'T_START_TIME': dt_time(hour = 9, minute = 30, second = 0, microsecond = 0),
            'T_END_TIME': datetime.strptime(market_close_time, '%H:%M:%S').time(),
            'LASSO_LAMBDA': lasso_lambda,
            'N_TICK_THRESHOLD': n_tick_threshold, # Tracey to notice
            'DATA_PATH': data_path,
            'N_HIST_DAY' : n_hist_day
        }

        for ticker in tickers:
            self.tickers[ticker] = VWAP(interval, ticker, self._params)
        
class VWAP(object):
    """
    a single VWAP object will track and predict one ticker
    """
    HALFTIME = timedelta(hours = 2)

    def __init__(self, interval, ticker, kwargs):
        
        if (interval % 5 != 0) or (7200 % interval != 0):
            raise ValueError('interval must be a multiple of 5 secs and can divide 2 hours')

        # ugly!
        # tickercsv = ticker + '.csv'
        # if not tickercsv in listdir(kwargs['DATA_PATH'] + (kwargs['TODAY'] - timedelta(days = 1)).strftime('%Y%m%d') ):
            # raise Exception('no data for %s' % ticker)

        dates = set(listdir(kwargs['DATA_PATH']))

        self.TODAY = kwargs['TODAY']
        # self.TODAY = datetime.strptime(today_for_test, "%Y-%m-%d")  # Tracey to notice
        self.T_START_TIME = kwargs['T_START_TIME']
        self.T_START_SEC = time.mktime(datetime.combine(self.TODAY, dt_time(hour = 9, minute = 30, second = 0, microsecond = 0) ).timetuple())
        # self.T_START_TIME = self.TODAY.replace(hour = 9, minute = 30, second = 0, microsecond = 0)
        self.T_END_TIME = kwargs['T_END_TIME']
        self.T_END_SECS = int((datetime.combine(self.TODAY,self.T_END_TIME) - datetime.combine(self.TODAY, self.T_START_TIME)).total_seconds())
        # self.T_END_TIME = self.TODAY.replace(hour = 15, minute = 00, second = 0, microsecond = 0)
        self.LASSO_LAMBDA = kwargs['LASSO_LAMBDA']
        self.N_TICK_THRESHOLD = kwargs['N_TICK_THRESHOLD'] # Tracey to notice
        self.DATA_PATH = kwargs['DATA_PATH']
        self.N4ROLLING = int(kwargs['N_HIST_DAY'] / 3)
        self.N4REGRESS = kwargs['N_HIST_DAY'] - self.N4ROLLING
        # self.DATA_PATH = './data_path/' # Tracey to notice
        self._interval = interval
        self._interval_timedelta = timedelta(seconds = self._interval)
        self._am_n_interval = int(self.HALFTIME.total_seconds() / self._interval_timedelta.total_seconds())
        self._n_interval = int(self._am_n_interval + ceil(( ( self.T_END_SECS - 60 * 60 * 3.5) / self._interval)))
        self._features_to_train = np.ones((self.N4REGRESS,3),dtype=float) # CA, M, L, A
        self._histo_volume = np.full((self.N4REGRESS, self._n_interval),0, dtype=float)  # historical trading volume        
        self._intraday_percentage = [1. / self._n_interval] * self._n_interval  # notice .sum() =self._n_interval
        # self._AR_pars = np.array([1,0],dtype =float) # (u and phi)
        self._AR_pars = [0., 1.] 
        self._CA_today = 0
        self._predicted_V = 0.
        self._is_V_predicted = 0
        self._last_update = 0
        self._iter = 0       
        self._datetime_index = ( [str(dt) for dt in datetime_range(self.T_START_TIME, 
                                dt_time(hour = 11, minute = 30, second = 0, 
                                microsecond = 0),timedelta(seconds = self._interval))] + 
                                [str(dt) for dt in datetime_range(dt_time(hour = 13, 
                                minute = 0, second = 0, microsecond = 0), 
                                self.T_END_TIME,timedelta(seconds = self._interval))])
        self._today_vol = [0.] * self._n_interval
        self._p_per = [1. / self._n_interval] * self._n_interval
        self._p_vol = [0] * self._n_interval
        self._cum_vol = 0
        self._VWAP_log = {}
        self._is_VWAP = 0

        tickercsv = ticker + '.csv'
        
        try:
            histo_date = self.TODAY
            past_days = 0
            x_output = np.concatenate((np.arange(0 + self._interval , 7200 + self._interval, self._interval), 
                                np.arange(12600 + self._interval, self.T_END_SECS, self._interval), np.array([self.T_END_SECS])), axis = 0)

            iter = 1
            
            while iter < (self.N4REGRESS + 1):

                if not bool(dates):
                    raise Exception('Insufficient historical data')

                histo_date = histo_date - timedelta(days = 1)
                past_days += 1

                # if histo_date.weekday() in set([5,6]):
                #    continue

                histo_date_str = histo_date.strftime("%Y%m%d")
                if histo_date_str not in dates:
                    continue
                dates.remove(histo_date_str)

                try:
                    dat = pd.read_csv(self.DATA_PATH + histo_date_str + '/' + tickercsv, header = 0)
                except Exception:
                    print 'Error in reading %s for %s, go to the previous day.' % (tickercsv, str(histo_date))
                    continue

                if dat.shape[0] < self.N_TICK_THRESHOLD:
                    print '%s in %s has few data for prediction' % (tickercsv, str(histo_date))
                    continue

                if past_days > 2 * self.N4REGRESS:
                    warnings.warn('Lack historical data. Time span of data for predicting intraday_volume of today has exceeded 2 times the desired days.'
                                    'We are using data %d days from today' % past_days) 

                try:
                    dat.Nano = dat.Nano / 1e9 - time.mktime(datetime.combine(histo_date, dt_time(hour = 9, minute = 30, second = 0, microsecond = 0)).timetuple())
                    tmp_Volume = np.array(dat.Volume)
                    dat.Volume = list(np.append(tmp_Volume[0], tmp_Volume[1:] - tmp_Volume[:-1]) )
                    dat = dat.as_matrix(columns = ['Nano', 'Volume']) # there will be Microsecond
                    datCA = dat[dat[:,0] < 0][:,1].sum() # Tracey to notice
                    if datCA < 1: # no data or no trade ?
                        continue
                    
                    self._features_to_train[self.N4REGRESS - iter,0] = datCA
                    dat = dat[dat[:,0] > 0]

                    # Tracey by reviewing the data from ctp finds it impossible
                    if any( 7200 < t < 7230 for t in dat[:,0]): # tracey_to_notice
                        dat = np.vstack( (dat[dat[:, 0] < 7200], [7200, dat[(dat[:,0] >= 7200)*(dat[:,0] < 7230),1].sum()], dat[dat[:, 0] > 7230]))
                    if any(t >= 198000 for t in dat[:,0]):
                        dat = np.vstack((dat[dat[:, 0] < 19800], [19800,dat[dat[:,0] >= 19800,1].sum()])) # tracey to notice

                    x_input = np.append(0, dat[:,0])
                    volume_cumsum = np.append(0,dat[:,1].cumsum())
                    # volume_cumsum = np.append(0, dat[:, 1]) # tracey to notice
                    y_interp = scipy.interpolate.interp1d(x_input,volume_cumsum) # ,interval)
                    intraday_volume = y_interp(x_output)
                    intraday_volume = np.append(intraday_volume[0],(intraday_volume[1:] - intraday_volume[:-1]))
                    self._histo_volume[self.N4REGRESS - iter] = intraday_volume # replace _histo_volume at row self.N4TREGRSS - iter
                except Exception:
                    print 'Error when read %s at %s, you may check its format' % (ticker, histo_date_str) 
                    continue

                iter += 1

            volume_sums = np.zeros(self.N4ROLLING, dtype=float)
            while iter < (self.N4REGRESS + self.N4ROLLING + 1):

                if not bool(dates):
                    raise Exception('Insufficient historical data')

                histo_date = histo_date - timedelta(days = 1)
                past_days += 1

                # if histo_date.weekday() in set([5,6]):
                #     continue

                histo_date_str = histo_date.strftime("%Y%m%d")
                if histo_date_str not in dates:
                    continue
                dates.remove(histo_date_str)

                try:
                    dat = pd.read_csv(self.DATA_PATH + histo_date_str + '/' + tickercsv, header = 0)
                except Exception:
                    print 'Error in reading %s for %s, go to the previous day.' % (tickercsv, str(histo_date)) 
                    continue     

                if dat.shape[0] < self.N_TICK_THRESHOLD:
                    print '%s in %s has few data for prediction' % (tickercsv, str(histo_date)) 
                    continue   

                if past_days > 3 * self.N4REGRESS:
                    warnings.warn('Lack efficacious historical data. Time span of data for predicting total trading volume of today has exceeded 3 times the desired days.')

                try:
                    dat.Nano = dat.Nano / 1e9 - time.mktime(datetime.combine(histo_date, dt_time(hour = 9, minute = 30, second = 0, microsecond = 0)).timetuple())                
                    tmp_Volume = np.array(dat.Volume)
                    dat.Volume = list(np.append(tmp_Volume[0], tmp_Volume[1:] - tmp_Volume[:-1]))            
                    dat = dat.as_matrix(columns = ['Nano', 'Volume']) # there will be Microsecond
                    volume_sums[self.N4REGRESS + self.N4ROLLING - iter] = dat[ (dat[:,0] > 0) * (dat[:,0] < self.T_END_SECS),1].sum()
                except Exception:
                    print 'Error when read %s at %s, you may check its format' % (ticker, histo_date_str) 
                    continue

                iter += 1

            # preparing sample for predicting today's total volume
            self.volume_to_train = self._histo_volume.sum(axis = 1)
            volume_sums = np.append(volume_sums, self.volume_to_train)
            self._features_to_train[:,1] = rolling_mean(volume_sums, self.N4ROLLING)
            self._features_to_train[:,2] = rolling_linear(volume_sums, self.N4ROLLING)

            # get intraday pattern and intialize intraday prediction
            intraday_mean = self._histo_volume.mean(axis = 0)
            self._p_vol[0] = int(intraday_mean[0])
            # self._intraday_percentage = list(np.divide(intraday_mean, intraday_mean.sum()) * self._n_interval)
            
            tmp = np.divide(intraday_mean, intraday_mean.sum()) * self._n_interval
            # print tmp # Tracey to notice
            if np.any(tmp < 0.1):
                warnings.warn('adjust intraday trading volume pattern for irregular data')
                tmp[tmp >= 0.1] = tmp[tmp > 0.1] * (self._n_interval - 0.1 * len(tmp[tmp < 0.1])) / sum(tmp[tmp >= 0.1])
                tmp[tmp < 0.1] = 0.1
            self._p_per = list(tmp / self._n_interval)
            self._intraday_percentage = list(tmp)
            # self._p_per = self._intraday_percentage / self._n_interval
            # self._p_per[0] = self._intraday_percentage[0] / self._n_interval
            
            self._VWAP_log[self._datetime_index[0]] = get_log(None, self._p_vol[0], self._p_per[0])         
            
            # compute AR
            arma = ARMA( self._histo_volume[-1] / self._intraday_percentage, order = (1,0))
            self._AR_pars = arma.fit().params.tolist()
            
            self._is_VWAP = 1
        except Exception:
            print 'Error to initialize %s, using TWAP' % ticker
            self._is_VWAP = 0

    def pred_V(self):
        
        if self._CA_today == 0:
            self._features_to_train[self.N4REGRESS,0] = self._features_to_train[-1,0].mean()
        else:
            self._features_to_train[self.N4REGRESS,0] = self._CA_today
        
        lm = Lasso(alpha = self.LASSO_LAMBDA)
        lm.fit(self._features_to_train[0:-1,:],self.volume_to_train)
        self._predicted_V = int(lm.predict(self._features_to_train[-1].reshape(1,-1))[0])
        if self._predicted_V < 0: 
            warnings.warn('We some how get a exceeding low volume prediction for today. We strongly urge you check your tick data.')
            self._predicted_V = 1 # Tr 1 acey to notice
        self._is_V_predicted = 1

    def get_log(self):
        return self._VWAP_log
    
    def get_predict(self, nano):
        
        sec_time =int(nano / 1e9 - self.T_START_SEC)

        if sec_time < 0 or sec_time >=  self.T_END_SECS:
            return 0.
        
        else:
            self.push_tick(nano, 0)
            return self._p_per[self._last_update]

    def push_tick(self, nano, cum_volume):
        
        if self._is_VWAP == 1:
            self.push_tick_1(nano, cum_volume)
        else:
            pass


    def push_tick_1(self, nano, cum_volume):

        volume = cum_volume - self._cum_vol

        self._cum_vol = cum_volume

        sec_time =int(nano / 1e9 - self.T_START_SEC)

        if sec_time < -900:
            print 'Illegal nano, too early for today'

        if sec_time > self.T_END_SECS:
            print 'Illegal nano, too late for today'

        if sec_time < 0: # if the trade happens in call auction
            self._CA_today += volume

        else:
            if not self._is_V_predicted:
                self.pred_V()

            iter = int(sec_time / self._interval)

            if iter > self._am_n_interval: # in the afternoon
                iter -= int(self._am_n_interval * 3 / 4)
            self._iter = iter
            
            if self._iter < self._n_interval: # time hasn't exceeded market close time

                if self._iter == self._last_update:
                    self._today_vol[self._iter] += volume                    

                elif  self._last_update < self._iter:
                    if self._last_update + 1 < self._iter:
                        warnings.warn('Over %d secs without receiving data' % self._interval)
                    
                    self._today_vol[(self._last_update + 1):(self._iter + 1)] = [int(round(a + b)) for a, b in zip(self._today_vol[(self._last_update + 1):(self._iter + 1)], [volume * s / sum(self._intraday_percentage[(self._last_update + 1):(self._iter + 1)]) for s in self._intraday_percentage[(self._last_update + 1):(self._iter + 1)]])]

                    for i in range(self._last_update, self._iter):
                        self._VWAP_log[self._datetime_index[i]] = get_log(self._today_vol[i], self._p_vol[i], self._p_per[i])
                        try: # bug
                            self._p_vol[i + 1] = int ((self._AR_pars[1] * (self._today_vol[i] / self._intraday_percentage[i] - self._AR_pars[0] ) + self._AR_pars[0] ) * self._intraday_percentage[i + 1])
                        except OverflowError:
                            print self._today_vol[i]
                            print self._intraday_percentage[i]
                            print self._intraday_percentage[i + 1]

                        if i + 2 < self._n_interval:
                            self._p_per[i + 1] = self._p_vol[i + 1] * (1 - sum(self._p_per[0:(i + 1)])) / (self._predicted_V * (1 - sum(self._intraday_percentage[0:(i + 1)])/ self._n_interval ))
                        else:
                            self._p_per[self._n_interval - 1] = 1 - sum(self._p_per[0:(self._n_interval - 1)])
                        self._VWAP_log[self._datetime_index[i + 1]] = get_log(None, self._p_vol[i + 1], self._p_per[i + 1])

                    self._last_update = self._iter
                    
                    self._p_per[(self._iter + 1):self._n_interval] = [ (1 - sum(self._p_per[0:(i + 1)])) / (sum(self._p_per[(self._iter + 1):self._n_interval])) * s for s in self._p_per[(self._iter + 1):self._n_interval]]
                    
            else: # self._iter has exceed the send of the market close time 
                self._iter = self._n_interval - 1

                if self._iter == self._last_update:
                    self._today_vol[self._iter] += volume
                
                if self._last_update < self._iter:
                    if self._last_update + 1 < self._iter:
                        warnings.warn('Over %d secs without receiving data' % self._interval)
                    
                    self._today_vol[(self._last_update + 1):(self._iter + 1)] = [a + b for a, b in zip(self._today_vol[(self._last_update + 1):(self._iter + 1)], [volume * s / sum(self._intraday_percentage[(self._last_update + 1):(self._iter + 1)]) for s in self._intraday_percentage[(self._last_update + 1):(self._iter + 1)]])]

                    for i in range(self._last_update, self._iter):
                        self._VWAP_log[self._datetime_index[i]] = get_log(self._today_vol[i], self._p_vol[i], self._p_per[i])
                        self._p_vol[i + 1] = int ((self._AR_pars[1] * (self._today_vol[i] / self._intraday_percentage[i] - self._AR_pars[0] ) + self._AR_pars[0] ) * self._intraday_percentage[i + 1])
                        
                        if i + 2 < self._n_interval:
                            self._p_per[i + 1] = self._p_vol[i + 1] * (1 - sum(self._p_per[0:(i + 1)])) / (self._predicted_V * (1 - sum(self._intraday_percentage[0:(i + 1)])/ self._n_interval ))
                        else:
                            self._p_per[self._n_interval - 1] = 1 - sum(self._p_per[0:(self._n_interval - 1)])
                        self._VWAP_log[self._datetime_index[i + 1]] = get_log(None, self._p_vol[i + 1], self._p_per[i + 1])

                    self._last_update = self._iter                    


if __name__ == "__main__":
    
    interval = 30

    today_for_test = '20170713'

    Tracey = VWAP_handler(interval, ['SH600884'], data_path = './data_path/', lasso_lambda = 812314)

    a = Tracey.tickers['SH600884']

    df = pd.read_csv(a.DATA_PATH + "20170713/SH600884.csv")

    for row in df.values.tolist():
        a.push_tick(row[0],row[1])
        if a._iter == 10:
            a.get_predict(row[0])

    print abs(np.array(a._p_per) - np.array(a._today_vol) / sum(np.array(a._today_vol))).sum()
    print abs(np.array([1. / len(a._p_per)] * len(a._p_per),dtype=float) - np.array(a._today_vol) / sum(np.array(a._today_vol))).sum()
    print abs(np.array(a._intraday_percentage ) / len(a._p_per) - np.array(a._today_vol) / sum(np.array(a._today_vol))).sum() 

    b = 1
        
    a.get_predict(df.values.tolist()[10][0])

    # a.get_predict()

    # print(a.get_predict())
