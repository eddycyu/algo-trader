# algo-trader

#### Usage (app_analyzer)
```
# fetch, analyzer and plot charts for default indices and equities
$ python app_analyzer

# fetch, analyze and plot charts 
# for indices (^SPX, ^NDQ, ^NDX) and equitites (AAPL, AMZN) 
# with earliest start date of 1980-01-01 and latest end date of today
$ python app_analyzer -i ^SPX -i ^NDQ -i ^NDX -i ^DJI -e AAPL -e AMZN --start=1980-01-01
```

#### Usage (app_predictor)
```
$ python app_predictor
```

#### Directories
By default, the following directories are created:
* data - location to store fetched time-series data files
* chart-technical - location to store generated technical analysis charts
* chart-prediction - location to store generated predictive analysis charts
* model - location to store generated models
* log - location to store log files