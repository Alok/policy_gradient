## Installation

Use [`pipenv`](https://github.com/kennethreitz/pipenv).

To install PyTorch, since they're not on PyPi, run
`pipenv install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl`
or whatever URL you need from the PyTorch website.

*Then*, run `pipenv run pip3 install --no-deps torchvision`.
