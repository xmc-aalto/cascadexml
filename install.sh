#!/bin/bash

pip3 install torch torchvision torchaudio
pip install transformers
pip install scipy
pip install scikit-learn
pip install cython
pip install tqdm
pip install nltk 
pip install six
pip install git+https://github.com/facebookresearch/fastText.git
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
# explicit fix for the issue https://github.com/kunaldahiya/pyxclib/issues/34  in pyxclib 
FILE_PATH="xclib/utils/_sparse.pyx"
sed -i \
    -e 's/dtype=np.int)/dtype='\''int'\'')/g' \
    -e 's/dtype=np.int,/dtype='\''int'\''/g' \
    -e 's/, np.int,/, '\''int'\'',/g' \
    -e 's/, np.float,/, '\''float'\'',/g' \
    "$FILE_PATH"
pip install .
cd ..
rm -rf pyxclib
