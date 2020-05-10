# For Tensorflow > 2.0 and Keras
- 2 types of settings:
    + on environment (your os system - linux, windows, mac)
        + It will run your script on multiple cores (CPUs - processors) 
        + 2 types:
            + type 0: taskset (some people said we should reset taskset due to the mixed settings of libraries like
             numpy, scikit-learn, scipy, pandas,....) ==> I tried but it is not working for tensorflow
            + type 1: os.environ['NAME'] = 'CORES'
            + type 2: os.sched_setaffinity(CORE_ID, {LIST_OF_CORE_IDS})
    + tensorflow operations
        + It will create multiple thread on single cores --> Take time to crease but faster if your program is long
        + tensorflow.config.threading.
            + set_intra_op_parallelism_threads(NUMBER_OF_CORES): matrix multiplication and reductions
            + set_inter_op_parallelism_threads(NUMBER_OF_CORES): number of threads used by independent non-blocking operations

- documents:
    + OS type 0: https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy/31370840#31370840
    + OS type 1: https://datascience.stackexchange.com/questions/22058/make-keras-run-on-multi-machine-multi-core-cpu-system
    + OS type 2: https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy/31370840#31370840
    + Tensorflow threading:
        + https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
        + https://www.tensorflow.org/api_docs/python/tf/config/threading/set_intra_op_parallelism_threads
    + Check processor (task) running on particular core: https://stackoverflow.com/questions/54902325/processes-running-on-a-particular-core


### 1. No tensorflow settings, environment - type 1 (1 cores or 2 cores) 

```code 
import everything_here
import tensorflow as tf

import os
# os.environ['MKL_NUM_THREADS'] = '2'           # 1
# os.environ['GOTO_NUM_THREADS'] = '2'          # 1
# os.environ['OMP_NUM_THREADS'] = '2'           # 1

## implement model below
....
print(time.time())
```

#### Single core OS
- Running on 5 cores (1 main core, 4 created thread cores) -- not 100% CPU usage (around 30%) -- 10.4 seconds
    + Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best
     performance.
    + ==> Default tensorflow parallel thread is 4
    
#### Multiple core OS
- Running on 9 cores (1 main core, 8 created thread cores) -- not 100% CPU usage (around 30%) -- 11.3 seconds 
    + Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best
     performance.
    + ==> Default tensorflow parallel thread is 8
    
==>>> Tensorflow auto tune number of threads created based on OS settings

 
### 2. Single thread tensorflow settings, environment - type 1 (1 cores or 2 cores) 

```code 
import everything_here
import tensorflow as tf

import os
# os.environ['MKL_NUM_THREADS'] = '1'           # 2  
# os.environ['GOTO_NUM_THREADS'] = '1'          # 2  
# os.environ['OMP_NUM_THREADS'] = '1'           # 2  

tf.config.threading.set_intra_op_parallelism_threads(1)     
tf.config.threading.set_inter_op_parallelism_threads(1)    

## implement model below
....
print(time.time())
```

#### Single core OS 
- Running on 2 cores (1 main core, 1 created thread cores) -- not 100% CPU usage (around 80%) -- 10.34 seconds
    + XLA service 0x558ffd7d01d0 initialized for platform Host (this does not guarantee that XLA will be used)
    + ==> better about running time and CPU usage
    
#### Multiple core OS
- Running on 2 cores (1 main core, 1 created thread cores) -- not 100% CPU usage (around 80%) -- 10.03 seconds 
    + XLA service 0x558ffd7d01d0 initialized for platform Host (this does not guarantee that XLA will be used)
    + ==> better about running time and CPU usage
    
==>>> If we dont use multiple threads in tensorflow, it wont have to take time to create and transfer data to
 different cores (each thread created will assigned to different core)
 


### 3. Multiple threads tensorflow settings, environment - type 1 (1 cores or 2 cores) 

```code 
import everything_here
import tensorflow as tf

import os
# os.environ['MKL_NUM_THREADS'] = '1'       # 2          
# os.environ['GOTO_NUM_THREADS'] = '1'      # 2    
# os.environ['OMP_NUM_THREADS'] = '1'       # 2    

tf.config.threading.set_intra_op_parallelism_threads(2)     
tf.config.threading.set_inter_op_parallelism_threads(2)     

## implement model below
....
print(time.time())
```

#### Single core OS


#### Multiple core OS
- Running on 3 cores (1 main core, 2 created thread cores) -- not 100% CPU usage (around 50%) -- 9.87 seconds 
    + XLA service 0x558ffd7d01d0 initialized for platform Host (this does not guarantee that XLA will be used)
    + ==> this case is ok but not what we want.


### 4. Single or Multiple threads tensorflow settings, environment - type 2 (affinity 1)

```code 
import everything_here
import tensorflow as tf

import os
os.sched_setaffinity(0, {1})

tf.config.threading.set_intra_op_parallelism_threads(2)     # 1 
tf.config.threading.set_inter_op_parallelism_threads(2)     # 1

## implement model below
....
print(time.time())
```


#### Single thread tensorflow
- Running on 1 core (1 main core) -- 100% CPU usage -- 8.8 seconds
    + The best so far, also this is what we want, because later we will use this with Multiprocessing Python 
      
#### Multiple threads tensorflow
- Running on 3 cores (1 main core, 2 created threads core) -- 50% CPU usage -- 11.4 seconds
    + Almost the worst, maybe it take time to create 2 created inside, meanwhile we assign main script with only 1 core.


# MultiProcessing Python combine with Tensorflow
- Assumption that I have 10 scripts python - 10 tasks which contain Keras (10 files). What is the best way to run 10
 scripts?
 
1. Run 10 files on 10 different screens?
    + Handy jobs too much (creating screens, activating environment, handling tensorflow multithreading,...)
    
2. Merge 10 files into single file, then using MultiProcessing python handle 10 tasks, then runs it in single screen?
    + All in one, we know how to handle tensorflow, now we only have to take care MultiProcessing Python, and also
     the assigment of affinity on cores.
    + Each task will run on single core, multi-threading tensorflow will execute in that single core, it will make
     each task run full 100 % CPU ===> Best performance. 
