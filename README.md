# icrp.jl
A Julia code for a submission "Network and interaction models for data with hierarchical granularity via fragmentation and coagulation"

## Running synthetic data experiments
Go to ``synthetic`` folder and follow the process below.

### Generating data
```
julia generate_data.jl
```
### Running HICRP sampler
```
julia run_pdgm_coag_icrp.jl
```
### Running independent SICRP samplers
```
julia run_icrp.jl
```
### Plotting
Execute ``plot_pdgm_coag_icrp.ipynb`` and ``plot_icrp.ipynb``

## Running wikipedia election network experiments
Go to ``wikivote`` folder and follow the process below.

### Generating data
Execute ``process.ipynb``

### Running HICRP sampler
```
julia run_pdgm_coag_icrp.jl
```
### Running independent SICRP samplers
```
julia run_icrp.jl
```
### Plotting
Execute ``plot_pdgm_coag_icrp.ipynb`` and ``plot_icrp.ipynb``



