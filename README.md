# State-of-the-art physics-based metaheuristics - multilayer perceptron for monthly streamflow time-series forecasting (Malaysia Team)

## Models
### Traditional 
1. MLP (Streamflow forecasting using artificial neural network and support vector machine models)
2. RNN (https://doi.org/10.1007/s11600-019-00330-1)
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
- Convergence errors of models in docs/results_of_paper/final_results_from_Thieu.xlsx
- Stability figures in the paper in docs/results_of_paper/images/...
