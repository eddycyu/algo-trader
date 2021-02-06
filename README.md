# algo-trader

#### Usage (app_train)
```
# fetch data, compute technical analysis, train network, output generate models to directory (model.yyyy-mm-dd)

$ python app_predictor --model=model.yyyy-mm-dd
```


#### Usage (app_predict)
```
# fetch data, compute technical analysis, plot technical analysis charts, 
# predict using models in directory (model.yyyy-mm-dd), and output processed data
$ python app_predict --model=model.yyyy-mm-dd

# fetch data, compute technical analysis, skip plotting of technical analysis charts, 
# predict using models in directory (model.yyyy-mm-dd), and output processed data
# for indices (^SPX, ^NDQ, ^NDX), stocks (AAPL, AMZN) and ETFs (SCHB, SCHX) 
# with earliest start date of 1980-01-01 and latest end date of today
$ python app_predict --no-plot --model=model.yyyy-mm-dd -i ^SPX -i ^NDQ -i ^NDX -i ^DJI -s AAPL -s AMZN -e SCHB -e SCHX --start=1980-01-01
```


#### Directories
By default, the following directories are created:
* data - location to store fetched time-series data files
* chart-technical - location to store generated technical analysis charts
* chart-prediction - location to store generated predictive analysis charts
* model - location to store generated models
* db - location to store processed data
* log - location to store log files