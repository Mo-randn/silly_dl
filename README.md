## silly_dl
Silly DL is a barebone deep learning "framework" to experiment and explore silly ideas

## Structure: 
<details> <summary>Repository structure</summary> <pre><code>silly_dl/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   └── train_mnist_dense.py
└── silly_dl/
    ├── __init__.py
    ├── datasets.py
    ├── layers.py
    ├── network.py</code></pre> </details>

## Installation

Clone the repository and install dependencies:

> git clone https://github.com/Mo-randn/silly_dl.git

> cd silly_dl

> python -m pip install -r requirements.txt

> python -m pip install -e .

## Usage


> python scripts/train_mnist_dense.py