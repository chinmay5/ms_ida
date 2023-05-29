#### Setup

#### Please create a virtual environment and include dependencies as specified in the `requirements.txt` file. 
If you are using pip, just execute
`pip install -r requirements.txt`

##### Please update the `config_1_y.ini` and `config_2_y.ini` files with location of the dataset.

Then please follow the steps to create the graphs for each of the patients

``` python -m graph_creation.create_per_patient_graph```

One can create the graph for a single patient and verify that things are working as expected by using

``` python -m graph_creation.create_small_graph```

Please note that all the intermediate results, model checkpoints and dataset storage/creation is governed by the config files. So, please
be very careful with the entries. Once the dataset is prepared, use

``` CUDA_VISIBLE_DEVICES=<GPU_ID>, CUBLAS_WORKSPACE_CONFIG=:16:8 python -m model_training.final_model_performance```

