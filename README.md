# Physics-Inspired Optimization Multi-Layer Perceptron for Rainfall Time Series Forecasting

## Models
### Traditional 
1. MLP
2. RNN 
3. LSTM 
4. GRU
### Evolutionary 
1. GA-MLP
2. DE-MLP
3. FPA-MLP
### Swarm 
1. PSO-MLP
2. WOA-MLP
3. GWO-MLP  (Grey Wolf Optimization)
4. SpaSA-MLP (Maybe this one is our proposed)
### Physics (Our proposed)
1. WDO-MLP
2. MVO-MLP
3. EO-MLP
4. NRO-MLP
5. HGSO-MLP

## How to run code
1. Run MLP model by: mlp.py 
2. Run Rnn-based model (RNN, LSTM) by: script_traditional_rrn_based.py (multiprocessing) 
3. All hybrid-MLP model by: script_hybrid_mlp.py (multiprocessing also)
4. Get the results table mean, std, cv by: get_table_results.py
5. Results saved in folder: history/results/ based on daily or weekly datatype
 
 
# Results in paper

1. Get table results using script: get_table_results.py
```code
All files stored in: history/results/

- daily_rainfall_22022020_mean.csv
- daily_rainfall_22022020_std.csv
- daily_rainfall_22022020_var.csv

- weekly_rainfall_22022020_mean.csv
- weekly_rainfall_22022020_std.csv
- weekly_rainfall_22022020_var.csv

Then using csv to create: 
    + daily_rainfall_22022020_final.csv
    + weekly_rainfall_22022020_final.csv
```

2. Drawing prediction
```code 
- Using csv to get the prediction files in folder: history/results/daily_rainfall_22022020/
- Prediction file based on the i-th run saved in: history/results/error/

- Now using the script: draw_prediction.py to create images in paper.
- There are 3 groups as followed:
    + Group 1: Truth, MLP, GA-MLP, PSO-MLP, MVO-MLP
    + Group 2: Truth, RNN, DE-MLP, WOA-MLP, EO-MLP
    + Group 3: Truth, LSTM, FPA-MLP, WDO-MLP, NRO-MLP
- Image files saved in: history/results/img/
```

3. Drawing convergence
```code 
- Using csv to get the error in training phase in folder: history/results/daily_rainfall_22022020/
- MSE error file based on the i-th run saved in: history/results/error/

- Now using the script: draw_error.py to create images in paper
- There are 2 groups as followed:
    + Group 1: MLP, RNN, GA-MLP, PSO-MLP, WDO-MLP, EO-MLP
    + Group 2: LSTM, DE-MLP, FPA-MLP, WOA-MLP, MVO-MLP, NRO-MLP
- Image files saved in: history/results/img/
```

4. Drawing stability
```code 
- Same as drawing prediction and convergence, this part is very handy.
- Create csv file from folder: history/results/daily_rainfall_22022020/
- The stability file saved in: history/results/error/stability_daily.csv [/stability_weekly.csv]

- Using plotly online to draw box plot. (https://plotly.com/)
```

