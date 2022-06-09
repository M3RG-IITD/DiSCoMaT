#!/bin/sh

conda create -n discomat python=3.7.9
conda activate discomat
conda install -y numpy==1.20.3 pandas==1.2.4 scikit-learn=0.23.2
conda install -y pytorch==1.7.0 cudatoolkit=10.1 -c pytorch

whl="torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl"
curl "https://data.pyg.org/whl/torch-1.7.0%2Bcu101/${whl}" --output $whl
pip install $whl
rm $whl

whl="dgl_cu101-0.7.1-cp37-cp37m-manylinux1_x86_64.whl"
curl "https://data.dgl.ai/wheels/dgl_cu101-0.7.1-cp37-cp37m-manylinux1_x86_64.whl" --output $whl
pip install $whl
rm $whl

pip install -r requirements.txt
