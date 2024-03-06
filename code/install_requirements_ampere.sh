#!/bin/sh

conda create -n discomat python=3.7.9
conda activate discomat
conda install -y numpy==1.20.3 pandas==1.2.4 scikit-learn=0.23.2
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

conda install -c conda-forge fairseq
conda install -c anaconda ujson
conda install -c conda-forge msgpack-python
conda install redis-py
conda install -c anaconda h5py

whl="torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl"
curl "https://data.pyg.org/whl/torch-1.7.0%2Bcu110/${whl}" --output $whl
pip install $whl
rm $whl

whl="dgl_cu110-0.7.1-cp37-cp37m-manylinux1_x86_64.whl"
curl "https://data.dgl.ai/wheels/dgl_cu110-0.7.1-cp37-cp37m-manylinux1_x86_64.whl" --output $whl
pip install $whl
rm $whl

pip install -r requirements.txt
