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


def getL(y):  # By linear regression predict the next value

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

    HALFTIME = timedelta(hours = 2)

    def __init__(self, interval, ticker, kwargs):
        '''
        construct a new 'VWAP' object.
        '''
        if (interval % 5 != 0) or (7200 % interval != 0):
            raise ValueError('interval must be a multiple of 5 secs and can divide 2 hours')
        
        # check tradeID is valid

        self.TODAY = kwargs['TODAY']
        # self.TODAY = datetime.strptime(today_for_test, "%Y-%m-%d")  # Tracey to notice
        self.T_START_TIME = kwargs['T_START_TIME']
        # self.T_START_TIME = self.TODAY.replace(hour = 9, minute = 30, second = 0, microsecond = 0)
        self.T_END_TIME = kwargs['T_END_TIME']
        self.T_END_TIME = self.TODAY.replace(hour = 15, minute = 00, second = 0, microsecond = 0)
        self.LASSO_LAMBDA = kwargs['LASSO_LAMBDA']
        self.N_TICK_THRESHOLD = kwargs['N_TICK_THRESHOLD']
        self.DATA_PATH = kwargs['DATA_PATH'] + ticker + '/'
        # self.DATA_PATH = './data_path/' # Tracey to notice
        self.files = set([ filename for filename in listdir(self.DATA_PATH) if filename.endswith( '.csv' ) ])
        self.interval = interval
        self.INTERVAL = timedelta(seconds = self.interval)
        self.nINTERVAL = 2 * int(self.HALFTIME / self.INTERVAL)
        self.pre_days = 0
        self.features_to_train = np.ones((11,3),dtype=float) # CA, M, L, A
        self.intraday_percentage = [1 / self.nINTERVAL] * self.nINTERVAL  # notice .sum() =self.nINTERVAL
        # self.AR_pars = np.array([1,0],dtype =float) # (u and phi)
        self.AR_pars = [0., 1.] 
        self.trad_volume = np.full((10,self.nINTERVAL),0, dtype=float)  # historical trading volume
        self.CAtoday = 0.
        self.predV = 0.
        self.is_V_predicted = 0
        self.last_update = 0
        self.iter = 0       
        self.datetime_index = ( [str(dt) for dt in datetime_range(self.T_START_TIME, 
                                self.T_START_TIME.replace(hour = 11, minute = 30, second = 0, 
                                microsecond = 0),timedelta(seconds = self.interval))] + 
                                [str(dt) for dt in datetime_range(self.T_START_TIME.replace(hour = 13, 
                                minute = 0, second = 0, microsecond = 0), 
                                self.T_END_TIME,timedelta(seconds = self.interval))])
        self.today_vol = [0.] * self.nINTERVAL
        self.predp = [0.] * self.nINTERVAL
        self.predv = [0] * self.nINTERVAL
        self.VWAP_log ={}
        
        history_date = self.TODAY
        x_output = np.append(np.arange(0 + self.interval , 7200 + self.interval, self.interval), 
                            np.arange(12600 + self.interval,19800 + self.interval,self.interval))
        iter = 1
        
        # get data for intraday prediction
        while iter < 11:

            if not bool(self.files):
                raise Exception('Insufficient historical data')
            
            history_date = history_date - timedelta(days = 1)
            self.pre_days += 1
            
            if history_date.weekday() in set([5,6]):
                continue
            
            # filename = str(history_date.strftime('%Y-%m-%d'))+'.csv' ## Tracey to notice
            filename = str(ticker) + str(history_date.strftime('%Y-%m-%d'))+'.csv' ## Tracey to notice
            if filename in self.files:
                self.files.remove(filename)
            else:
                continue


            try:
                dat = pd.read_csv(self.DATA_PATH + filename)
            except Exception:
                print('Error in reading ' + filename + ', go to the previous day.')
                continue

            if dat.shape[0] < self.N_TICK_THRESHOLD:
                print('File ' + filename + ' has few data for prediction')
                continue

            print(filename + 'intra')
            if self.pre_days > 20:
                warnings.warn('Lack historical data. Time span of data for predicting intraday_volume of today has exceeded 20 days.')        
            
            try:
                dat.columns = ['DateTime','Volume'] # there will be Microsecond
                dat.DateTime = [datetime.strptime(str(history_date.strftime('%Y-%m-%d')) + ' ' + dt, "%Y-%m-%d %H:%M:%S") for dt in dat.DateTime]
                
                # datetime to time difference
                self.H_START_TIME = history_date.replace(hour = 9, minute = 30, second = 0, microsecond = 0)            
                dat['TimeStamp'] = [(dt - self.H_START_TIME).total_seconds() for dt in dat.DateTime]
                dat = dat.as_matrix(columns = ['TimeStamp','Volume'])
                
                datCA = dat[dat[:,0] < 0]
                self.features_to_train[10 - iter,0] = datCA[:,1].sum()
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
                self.trad_volume[10 - iter] = intraday_volume
            except Exception:
                print('Error when read file '+ filename + ', you may check its format')
                continue

            iter += 1
            # print('Done'+str(iter))



        iter = 11

        # get data for roll_mean and roll_linear
        volume_sums = np.zeros(5,dtype=float)
        history_date = self.TODAY -  timedelta(days = self.pre_days)
        while iter < 16:

            if not bool(self.files):
                raise Exception('Insufficient historical data')
            
            history_date = history_date - timedelta(days = 1)
            self.pre_days += 1
            
            if history_date.weekday() in set([5,6]):
                continue
            
            # filename = str(history_date.strftime('%Y-%m-%d'))+'.csv' ## Tracey to notice
            filename = str(ticker) + str(history_date.strftime('%Y-%m-%d'))+'.csv' ## Tracey to notice
            if filename in self.files:
                self.files.remove(filename)
            else:
                continue

            try:
                dat = pd.read_csv(self.DATA_PATH+filename)
            except Exception:
                print('Error in reading ' + filename + ', go to the previous day.')
                continue

            # if dat.shape[0] < self.N_TICK_THRESHOLD:
            #     print('File ' + filename + ' has few data for prediction')
            #     continue
            print(filename)
            if self.pre_days > 30:
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
        self.volume_to_train = self.trad_volume.sum(axis = 1)
        volume_sums = np.append(volume_sums, self.volume_to_train)
        self.features_to_train[:,1] = rolling_mean(volume_sums)
        self.features_to_train[:,2] = rolling_linear(volume_sums)

        # get intraday pattern and intialize intraday prediction
        intraday_mean = self.trad_volume.mean(axis = 0)
        self.predv[0] = float(intraday_mean[0])
        self.intraday_percentage = list(np.divide(intraday_mean, intraday_mean.sum()) * self.nINTERVAL)
        self.predp[0] = self.intraday_percentage[0] / self.nINTERVAL
        self.VWAP_log[self.datetime_index[0]] = get_log(None, self.predv[0], self.predp[0])         
        
        # compute AR
        arma = ARMA(self.trad_volume[-1]/self.intraday_percentage, order = (1,0))
        self.AR_pars = arma.fit().params.tolist()

    def pred_V(self):
        if self.CAtoday == 0:
            self.features_to_train[10,0] = self.features_to_train[:,0].sum()
        else:
            self.features_to_train[10,0] = self.CAtoday
        lm = Lasso(alpha = self.LASSO_LAMBDA)
        lm.fit(self.features_to_train[0:-1,:],self.volume_to_train)
        self.predV = lm.predict(self.features_to_train[-1].reshape(1,-1))[0]
        if self.predV < 0:
            self.predV = 1 # Tracey to notice
        self.is_V_predicted = 1
        print('finish: pred_V')

    def push_tick(self, date_time, volume):
        if date_time < self.T_START_TIME:
            self.CAtoday += volume
        elif date_time < self.T_END_TIME: 
            if not self.is_V_predicted:
                self.pred_V()
            iter = int((date_time - self.T_START_TIME) / self.INTERVAL)
            # if iter >= int(self.nINTERVAL * 11 / 8):
            #     iter = int(self.nINTERVAL * 11 / 8) - 1
            if iter > (self.nINTERVAL / 2):
                iter -= int(self.nINTERVAL * 3 / 8)
            self.today_vol[iter] += volume
            self.iter = iter
            if self.iter == self.last_update:
                pass
            elif self.iter - self.last_update == 1:
                self.VWAP_log[self.datetime_index[self.last_update]] = get_log(self.today_vol[self.last_update], self.predv[self.last_update], self.predp[self.last_update])
                self.predv[self.iter] = int ((self.AR_pars[1] * (self.today_vol[self.last_update] / self.intraday_percentage[self.last_update] - self.AR_pars[0] ) + self.AR_pars[0] ) * self.intraday_percentage[self.iter])
                if self.iter < (self.nINTERVAL - 1):
                    self.predp[self.iter] = self.predv[self.iter] * (1 - sum(self.predp[0:self.iter])) / (self.predV * (1 - sum(self.intraday_percentage[0:self.iter])/ self.nINTERVAL ))
                else:
                    self.predp[self.nINTERVAL - 1] = 1 - sum(self.predp[0:(self.nINTERVAL - 1)])
                self.VWAP_log[self.datetime_index[self.iter]] = get_log(None, self.predv[self.iter], self.predp[self.iter])
                self.last_update = self.iter
            elif self.iter - self.last_update > 1:
                warnings.warn('Over %d secs without receiving data' % self.interval)
                self.today_vol[iter] =+ volume
                self.today_vol[self.last_update:self.iter] = [a + b for a, b in zip(self.today_vol[self.last_update:self.iter], [volume * s / sum(self.intraday_percentage[self.last_update:self.iter]) for s in self.intraday_percentage[self.last_update:self.iter]])]
                for i in range(self.last_update, self.iter):
                    self.VWAP_log[self.datetime_index[i]] = get_log(self.today_vol[i], self.predv[i], self.predp[i])
                    self.predv[i + 1] = int ((self.AR_pars[1] * (self.today_vol[i] / self.intraday_percentage[i] - self.AR_pars[0] ) + self.AR_pars[0] ) * self.intraday_percentage[i + 1]) 
                    if i + 1 < (self.nINTERVAL - 1):
                        self.predp[i + 1] = self.predv[i + 1] * (1 - sum(self.predp[0:(i + 1)])) / (self.predV * (1 - sum(self.intraday_percentage[0:(i + 1)])/ self.nINTERVAL ))
                    else:
                        self.predp[self.nINTERVAL - 1] = 1 - sum(self.predp[0:(self.nINTERVAL - 1)])
                    self.VWAP_log[self.datetime_index[i + 1]] = get_log(None, self.predv[i + 1], self.predp[i + 1])
                self.last_update = self.iter                    
            else: # when self.iter < self.last_update, we only update real volume
                pass
        else:
            self.today_vol[self.nINTERVAL - 1] += volume 
            self.VWAP_log[self.datetime_index[self.nINTERVAL - 1]] = get_log(self.today_vol[self.nINTERVAL - 1], self.predv[self.nINTERVAL - 1], self.predp[self.nINTERVAL - 1])            

    def get_predict(self):
        return(self.VWAP_log)

class VWAPs(object):

    """
    Examples:
    VWAPs(30,['SH00019','SH000018'], lasso_lambda = 812314)
    """

    def __init__(self, interval, tickers, lasso_lambda = 812314, 
                n_tick_threshold = 1000, data_path = './data_path/'):

        self.tickers = {}
        self.params = {
            'TODAY': datetime.today(),
            'T_START_TIME': datetime.today().replace(hour = 9, minute = 30, second = 0, microsecond = 0),
            'T_END_TIME': datetime.today().replace(hour = 15, minute = 0, second = 0, microsecond = 0),
            'LASSO_LAMBDA': lasso_lambda,
            'N_TICK_THRESHOLD': n_tick_threshold,
            'DATA_PATH': data_path  
        }
        for ticker in tickers:
            self.tickers[ticker] = VWAP(interval, ticker, self.params)

a = VWAPs(300, ["SH000019"])

print(a.tickers["SH000019"].get_predict())
            
