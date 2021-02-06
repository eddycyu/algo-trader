# value constants
CALENDAR_DAYS = 365
TRADING_DAYS_YEAR = 252
TRADING_DAYS_MONTH = 22

# directory constants
LOG_DIR = "log"
DATA_DIR = "data"
MODEL_DIR = "model"
DB_SYMBOL_DIR = "db"
DB_PREDICT_DIR = "predict"
DB_PERF_DIR = "perf"
CHART_TRAIN_DIR = "chart-train"
CHART_PREDICT_DIR = "chart-predict"
CHART_TA_DIR = "chart-technical"

# column constants
CLOSE = "u_close"
OPEN = "Open"
LOW = "Low"
HIGH = "High"
VOLUME = "Volume"
PREV_CLOSE = "u_prev_close"
CHANGE = "u_change"
CHANGE_PC = "u_change_pc"
OPEN_PREV_CLOSE = "u_open_prev_close"
OPEN_PREV_CLOSE_PC = "u_open_prev_close_pc"
R52_WK_LOW = "u_52_wk_low"
R52_WK_HIGH = "u_52_wk_high"
CLOSE_ABOVE_52_WK_LOW = "u_close_above_52_wk_low"
CLOSE_BELOW_52_WK_HIGH = "u_close_below_52_wk_high"
CLOSE_ABOVE_52_WK_LOW_PC = "u_close_above_52_wk_low_pc"
CLOSE_BELOW_52_WK_HIGH_PC = "u_close_below_52_wk_high_pc"
SMA = "u_sma"
SMA_GOLDEN_CROSS = "u_sma_golden"
SMA_DEATH_CROSS = "u_sma_death"
EMA = "u_ema"
EMA_GOLDEN_CROSS = "u_ema_golden"
EMA_DEATH_CROSS = "u_ema_death"
ADTV = "u_adtv"
BB = "u_bb"
MACD = "u_macd"
MACD_SIGNAL = "u_macd_signal"
MACD_HISTOGRAM = "u_macd_histogram"
RSI_AVG_GAIN = "u_rsi_avg_gain"
RSI_AVG_LOSS = "u_rsi_avg_loss"
RSI_RS = "u_rsi_rs"
RSI = "u_rsi"
SHARPE = "u_sharpe"
TRAILING_RETURN = "u_trailing_return"
ANNUALIZED_RETURN = "u_annualized_return"
MAX_DD = "u_mdd"

# index symbols
#   ^SPX/^GSPC = S&P500 = cap-weighted index of the 500 largest U.S. publicly traded companies
#   ^TWSE = cap-weighted index of all listed common shares traded on the Taiwan Stock Exchange
#   ^KOSPI = cap-weighted index of all listed common shares traded on the Korea Exchange
#   ^NKX = price-weighted index of top 225 blue-chip companies traded on the Tokyo Stock Exchange
#   ^HSI = cap-weighted index of the largest companies on the Hong Kong Exchange
#   ^STI = cap-weighted index of top 30 companies on the Singapore Exchange
#   ^SHC = cap-weighted index of all stocks (A-shares and B-shares) traded on the Shanghai Stock Exchange
#   ^SHBS = cap-weighted index of all B-shares traded on the Shanghai Stock Exchange
SYMBOLS_INDICES_NON_US = ("^TWSE", "^KOSPI", "^NKX", "^HSI", "^STI", "^SHC", "^SHBS")
SYMBOLS_INDICES = ("^GSPC", "^IXIC", "^NDX", "^DJI", "^RUT", "^VIX")

# stock symbols
SYMBOLS_STOCKS = (
    "AAPL", "AMZN", "FB", "GOOGL", "GOOG", "MSFT", "NFLX", "TSLA", "CRM", "WDAY",
    "BABA", "TCEHY", "NIO", "NOW", "SNOW", "PLTR",
    "SQ", "TWOU", "Z", "TREE", "WORK", "TDOC", "PSTG", "XLNX", "VCYT", "SPLK", "PINS", "TWLO", "SE", "TSM", "ICE",
    "DOCU", "NVDA", "ONVO", "NVTA", "CRSP", "ROKU", "ILMN", "CGEN", "EDIT", "PYPL", "MELI", "DE", "PRLB", "JD", "FLIR",
    "ARCT", "PACB", "TWST", "CDNA", "IOVA", "EXAS", "CLLS", "SPOT", "GBTC",
    "SVFAU", "PSTH", "IPOD", "IPOE", "IPOF", "SVAC"
#   "SVFA", "SVFAW", "LDHAU", "SVFBU", "SVFCU"
)
# ETF symbols
SYMBOLS_ETFS = (
    "SCHB", "SCHX", "SCHG", "SCHA", "SCHF", "SCHE", "SCHD", "SCHH", "SCHZ",
    "VTI", "VOO", "VTV", "VEA", "VWO", "VGT", "QQQ", "SPY", "GLD", "IWF", "AGG",
    "ARKK", "ARKW", "ARKF", "ARKG", "ARKQ",
    "SOXL", "TECL", "TQQQ",
    "FXI", "ROM", "IPO",
    "XLY", "XLK", "XLRE", "XLC", "XLF", "XLB", "XLI", "XLE", "XLU", "XLV", "XLP"
)

# settings for model training
# < 100 hidden nodes = 500 epochs
# 100~140 hidden nodes = 200 epochs
# > 140 hidden nodes = 100 epochs
TRAIN_SETTINGS = {
    "^SPX": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "^NDQ": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "^NDX": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "^DJI": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "^TWSE": {"epochs": 200, "steps_in": 20, "steps_out": 1},
    "^KOSPI": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "^NKX": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "^HSI": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "^STI": {"epochs": 200, "steps_in": 20, "steps_out": 1},
    "^SHC": {"epochs": 200, "steps_in": 20, "steps_out": 1},
    "^SHBS": {"epochs": 500, "steps_in": 20, "steps_out": 1},

    "AAPL": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "AMZN": {"epochs": 200, "steps_in": 20, "steps_out": 1},
    "FB": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "GOOGL": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "GOOG": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "MSFT": {"epochs": 100, "steps_in": 20, "steps_out": 1},
    "NFLX": {"epochs": 200, "steps_in": 20, "steps_out": 1},
    "TSLA": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "CRM": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "WDAY": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "BABA": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "TCTZF": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "NIO": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "NOW": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SNOW": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "PLTR": {"epochs": 500, "steps_in": 20, "steps_out": 1},

    "SCHB": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SCHX": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SCHG": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SCHA": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SCHF": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SCHE": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SCHD": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SCHH": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SCHZ": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "VTI": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "VOO": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "VTV": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "VEA": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "VWO": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "QQQ": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "SPY": {"epochs": 200, "steps_in": 20, "steps_out": 1},
    "GLD": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "IWF": {"epochs": 500, "steps_in": 20, "steps_out": 1},
    "AGG": {"epochs": 500, "steps_in": 20, "steps_out": 1},
}
