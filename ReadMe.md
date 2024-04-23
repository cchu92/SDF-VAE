# VAE implicit parameterized 3d structure dataset represented by signed distance field(SDF-VAE)

## prepared the data
Each cell in the dataset is saved as a  `.npy` named according to its index(e.g.`cell_index.npy`). These files are located at:

```
./src/data/npy_file
```

## Initial Setup
1. Ensure the configuration settings are properly set in `pre_dataset_config.json`
2. Use `dataset.py` to read all SDF files and prepare the training and test datasets.


## Train the model

1. Ensure the configuration settings of the trainig model `config_SDF_VAE.json` 
2. The structure of the Convolutional Neural Network (CNN) depends on the resolution of the SDFs. By default, the resolution is set to $50^3$, If you change the resolution, you must also adjust the parameters of the last layers to guarantee the same output resolution.
3. hight recomend GPU mahinve to run the trainig code `main_train_sdf_vae.py`, after training, reuse the model by `reuse_model.py`



