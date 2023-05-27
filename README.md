# DeepIM

The code has been tested under Pytorch 1.8
```
docker pull ghcr.io/anty-filidor/mlaudio:alpha
<enter docker and forward host's GPUs>
conda env create -f conda_env.yml
```

To test the code, simply run the following command  
```
cd <DeepIM>
python genim.py -d <datasetname>
```
Datasets names are 'jazz', 'power_grid', 'cora_ml', 'netscience', 'random5'
Run the following code to see available parameters that can be passed in:  
```
python genim.py -h
```

## Notes for the seminary

### Variational Auto-Encoders
- explained extra fast: https://www.youtube.com/watch?v=sV2FOdGqlX0
- explained with better intuition: https://www.youtube.com/watch?v=9zKuYvjFFS8
- tutorial: https://towardsdatascience.com/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2

### Graph Attention Networks
- comprehensive explanation: https://www.baeldung.com/cs/graph-attention-networks
- explained very fast: https://www.youtube.com/watch?v=SnRfBfXwLuY
- spectral GAT (used in this repo): https://github.com/SwiftieH/SpGAT

### NDLIB
- tutoaial: https://ndlib.readthedocs.io/en/latest/tutorial.html

## The paper:
- preprint: https://arxiv.org/abs/2305.02200 (also in the repo)
- ICML page: https://icml.cc/ (will be published soon)
