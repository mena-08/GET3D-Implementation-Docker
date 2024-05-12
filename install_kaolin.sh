git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin

export IGNORE_TORCH_VER=1
export CUB_HOME=/usr/local/cuda-*/include/

python setup.py develop

#if everything went ok, shouldn't be any errors
python -c "import kaolin; print(kaolin.__version__)"
