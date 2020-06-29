# Physics-Inspired Optimization Multi-Layer Perceptron for Rainfall Time Series Forecasting

## Models
### Traditional 
1. MLP (Streamflow forecasting using artificial neural network and support vector machine models)
2. RNN (https://doi.org/10.1007/s11600-019-00330-1)
3. LSTM (https://doi.org/10.1016/j.jhydrol.2019.124296)
4. GRU (https://doi.org/10.1007/978-3-030-32388-2_44)
### Evolutionary 
1. GA-MLP (Predicting monsoon floods in rivers embedding wavelet transform, genetic algorithm and neural network)
2. DE-MLP (Differential evolution algorithms applied to neural network training suffer from stagnation)
3. FPA-MLP (Not found yet, you can check again)
### Swarm 
1. PSO-MLP (Hybrid particle swarm and neural network approach for streamflow forecasting)
2. WOA-MLP (Annual Rainfall Forecasting Using Hybrid Artificial Intelligence Model: Integration of Multilayer Perceptron with Whale Optimization Algorithm)
3. GWO-MLP  (Improving artificial intelligence models accuracy for monthly streamflow forecasting using grey Wolf optimization (GWO) algorithm)
4. SpaSA-MLP (No one - A novel swarm intelligence optimization approach: sparrow search algorithm)
### Physics (Our proposed)
1. WDO-MLP (Wind Driven Optimization - No one use it yet, but this algorithm is old - 2010)
2. MVO-MLP (Multi-Verse Optimization - No one use it yet, 2016)
3. EO-MLP (Equilibrium optimizer: A novel optimization algorithm - No one use it yet, 2019)
4. NRO-MLP (Nuclear Reaction Optimization: A novel and powerful physics-based algorithm for global optimization - No
 one use it yet, 2019)
5. HGSO-MLP (Henry gas solubility optimization: A novel physics-based algorithm, 2019)

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

