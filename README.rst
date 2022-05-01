===============================
Adolphs Lab, eye tracking tools
===============================

Eye tracking data analysis tools used in the Adolphs Lab.


clone/download and installation
===============================

suggested way: clone the repository, set up a conda environment, and install the development version in editable mode via:

.. code-block:: bash

    git clone https://github.com/adolphslab/adolphslab_eyetracking
    cd adolphslab_eyetracking
    conda env create --file make_env.yml
    conda activate eyetracking_env
    pip install -e .
    

alternatively, pip installation -- but some dependencies could be problematic:

.. code-block:: bash

    git clone https://github.com/adolphslab/adolphslab_eyetracking
    cd adolphslab_eyetracking
    pip install -e .


