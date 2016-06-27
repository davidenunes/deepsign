# DeepSign Project
A proof of concept framework for unsupervised learning of language patterns, models, and vector representations (embeddings).

## DeepSign Virtual Environment
This project depends on the latest [tensorflow](https://www.tensorflow.org/) installation (Python 3.5) along with other
dependencies listed in the `requirements.txt` file provided in this package.

A complete installation would be something as follows:

### Creating a virtualenv
* Make sure python 3.5 is installed.
* Create a `virtualenv` using _virtualenv_ or _[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)_

 ```bash

 export WORKON_HOME=~/dev/envs
 mkdir -p $WORKON_HOME
 source /usr/local/bin/virtualenvwrapper.sh
 # or
 source /usr/bin/virtualenvwrapper.sh
 mkvirtualenv --python=/usr/bin/python3 deepsign
 ```

 **Note**: To access the environment anytime just run `workon deepsign`
### Install **TensorFlow**
To install **TensorFlow** on the newly setup `virtualenv` just run the following:
 ```bash
 workon deepsign
 pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp35-cp35m-linux_x86_64.whl
 ```

 This is the _CPU only_ version, if you need other versions see [TensorFlow installation](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#virtualenv-installation). This should take care of the tensorflow dependency.

 ### Install TensorX
 This is my library that simplifies the development of tensorflow models (attemps to reduce the verbose) when creating typical
 neural network computation graphs.

 Either use some _IDE_ like [PyCharm](https://www.jetbrains.com/pycharm/) to manage these dependencies separately
 by downloading the latest version of **TensorX** from [GitHub](https://github.com/davidenunes/tensorx).

 Alternatively, install it in the current environment using:

 ```bash
 workon deepsign
 pip3 install --upgrade git+https://github.com/davidelnunes/tensorx.git
 ```

 **Note**: _TensorX_ also depends on _TensorFlow_, so this assumes tensorflow is already installed in the environment.

### Installing the Remaining Dependencies
Take the `requirements.txt` file provided and run the following

```bash
workon deepsign
pip3 install -r /path/to/requirements.txt
```

### Done