Installation
Install 🤗 Transformers for whichever deep learning library you’re working with, setup your cache, and optionally configure 🤗 Transformers to run offline.

🤗 Transformers is tested on Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, and Flax. Follow the installation instructions below for the deep learning library you are using:

PyTorch[https://pytorch.org/get-started/locally/] installation instructions.
TensorFlow 2.0[https://www.tensorflow.org/install/pip] installation instructions.
Flax[https://flax.readthedocs.io/en/latest/] installation instructions.

Install with pip
You should install 🤗 Transformers in a virtual environment. 
If you’re unfamiliar with Python virtual environments, 
take a look at this guide.[https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/]


A virtual environment makes it easier to manage different projects, and avoid compatibility issues between dependencies.

Start by creating a virtual environment in your project directory:
"python -m venv .env"
On Linux and MacOs:"python3 -m venv .venv"
on Windows:"py -m venv .venv"

Activate the virtual environment. On Linux and MacOs:
"source .env/bin/activate"

Activate Virtual environment on Windows ".env/Scripts/activate"

Now you’re ready to install 🤗 Transformers with the following command:
"pip install transformers"

For CPU-support only, 
you can conveniently install 🤗 Transformers and a deep learning library in one line. 
For example, install 🤗 Transformers and PyTorch with:
"pip install 'transformers[torch]'"

🤗 Transformers and TensorFlow 2.0:
"pip install 'transformers[tf-cpu]'"

Now Run .py files and enjoy !! Feel free to Edit and Learn
