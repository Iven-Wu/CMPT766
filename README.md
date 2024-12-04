# From Pixels to Motion: Universal 3D Animal Animation from Monocular Video
This repository contains the implementation for CMPT 766 final project.
Group member: Yiming Zhang (301354482), Yuefan Wu (301559392), Jiacheng Chen (301324027)

## Environmental setup

```
conda env create -f env.yml
conda activate env

# install lietorch
git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch
python setup.py install
cd ..

# install clip
pip install git+https://github.com/openai/CLIP.git

# install softras
# to compile for different GPU arch, see https://discuss.pytorch.org/t/compiling-pytorch-on-devices-with-different-cuda-capability/106409
pip install -e softras
```


## How to run

```
# Optimize one animal
python train.py --config_file config/synthetic.yaml
```