# code_inflow_rainfall

### Paper Name
1. Rainfall Time-series Prediction Using Long-Short Term Memory With Cutting-Edge Physics-Inpired Meta-heuristics.
2. The state-of-the-art physics-inspired meta-heuristics for training Long-short term memory network for rainfall time
-series prediction.


# Models
### Traditional 
1. MLP
2. RNN 
3. LSTM 
### Evolutionary 
1. GA-MLP
2. DE-MLP
3. FPA-MLP
### Swarm 
1. PSO-MLP
2. WOA-MLP
### Physics (Our proposed)
1. WDO-MLP
2. MVO-MLP
3. EO-MLP
4. NRO-MLP

# How to run code
1. Run MLP model by: mlp.py 
2. Run Rnn-based model (RNN, LSTM) by: script_traditional_rrn_based.py (multiprocessing) 
3. All hybrid-MLP model by: script_hybrid_mlp.py (multiprocessing also)
4. Get the results table mean, std, cv by: get_table_results.py
5. Results saved in folder: history/results/ based on daily or weekly datatype
 