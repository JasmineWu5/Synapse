# Synapse

### Environment Setup

To set up the environment, use **python venv** or **Miniconda** to create a virtual environment, 
it's recommended to use python 3.11, other versions may work but are not tested.

Taking **python venv** as an example:

```bash
python3.11 -m venv synapse_venv
```

Then activate the virtual environment:

```bash
source synapse_venv/bin/activate
```

Install Synapse in editable mode:
```bash
git clone git@github.com:Shudong-Wang/Synapse.git
cd Synapse
pip install -e .
```

I'm considering publishing this package to PyPI, but it will take some time.

### Usage

#### For HHML analysis
First, you need to convert your CAF ntuple to the format accepted by Synapse.
```bash
cp Synapse/src/synapse/configs/convert_config.yaml /path/to/your/working/directory
# After editing convert_config.yaml
synapse-convert-hhml -c convert_config.yaml
```

Then you can run the Synapse training:
```bash
cp Synapse/src/synapse/configs/run_config.yaml /path/to/your/working/directory
cp Synapse/src/synapse/configs/data_config.yaml /path/to/your/working/directory
cp Synapse/src/synapse/configs/model_config.yaml /path/to/your/working/directory
# After properly editing run_config.yaml, data_config.yaml, and model_config.yaml
synapse -d data_config.yaml -m model_config.yaml -r run_config.yaml
```
