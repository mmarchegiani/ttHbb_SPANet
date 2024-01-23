# ttHbb_SPANet
Repository for development of a signal vs background classifier in the ttH(bb) analysis based on SPANet.

## Running SPANet within the `cmsml` docker container

In order to use the SPANet package we use the prebuilt **apptainer** image for machine learning applications in CMS, [`cmsml`](https://hub.docker.com/r/cmsml/cmsml).

First we activate the apptainer environment on **lxplus** with the following command:

```bash
apptainer shell -B /afs -B /eos --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest
```

All the common packages for machine learning applications are now available in a singularity shell.
We can proceed by installing `SPANet` in a virtual environment inside this Docker container:
```bash
# Clone locally the SPANet repository
git clone git@github.com:Alexanders101/SPANet.git
cd SPANet

# Create a local virtual environment using the packages defined in the apptainer image
python -m venv --system-site-packages myenv

# Activate the environment
source myenv/bin/activate

# Install in EDITABLE mode
pip install -e .
```

The next time the user enters in the apptainer the virtual environment needs to be activated.
```bash
#Enter the image
apptainer shell -B /afs -B /eos --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest

# Activate the virtual environment
cd SPANet
source myenv/bin/activate
```

To check that SPANet is correctly installed in the environment, run the following command:
```bash
python -m spanet.train --help
```
