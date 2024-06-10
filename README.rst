Continual Resilient (CoRe) Optimizer
====================================

Introduction
------------

The module core_optimizer provides a PyTorch implementation of the Continual Resilient (CoRe) optimizer.
The CoRe optimizer is a first-order gradient-based optimizer for stochastic and deterministic iterative optimizations.
It applies weight-specific learning rate adaption depending on the optimization progress.
Its algorithm combines Adam- and RPROP-like step size updates and employs weight regularization and decay.

Installation
------------

The module core_optimizer can be installed using pip once the repository has been cloned.

.. code-block:: bash

   git clone <core_optimizer-repository>
   cd <core_optimizer-repository>
   python3 -m pip install .

A non super user can install the package using a virtual environment or the ``--user`` flag.
If there is no space left on device for TMPDIR, one can use ``TMPDIR=<PATH>`` in front of python3,
with <PATH> being a directory with more space for temporary files.

Usage
-----

The CoRe optimizer can be applied in the same way as the optimizers in torch.optim.
The only exception is the import of the optimizer.

.. code-block:: python

   from core_optimizer import CoRe

   optimizer = CoRe(...)   # dots are placeholders for model parameters and optimizer hyperparameters

For comparison the following code block shows the usage of the Adam optimizer from torch.optim.

.. code-block:: python

   from torch.optim import Adam

   optimizer = Adam(...)   # dots are placeholders for model parameters and optimizer hyperparameters

The minimal workflow of training a PyTorch model is shown in the next code block.
Examples which are ready to go can be found in test_core_optimizer.py.

.. code-block:: python

   import torch
   from core_optimizer import CoRe

   # set up input and expected outputs for training
   inputs = torch.tensor(...)   # dots are placeholders for input data
   labels = torch.tensor(...)   # dots are placeholders for expected output data

   # define loss function, model, and optimizer
   loss_fn = torch.nn.modules.loss.TorchLoss()   # TorchLoss is a placeholder for any Torch loss function
   model = TorchModel()   # TorchModel is a placeholder for any Torch model
   optimizer = CoRe(model.parameters(), step_sizes=(1e-6, 1e-2))   # define CoRe optimizer

   # run training steps
   for i in range(...):   # dots are a placeholder for number of steps
       optimizer.zero_grad()   # zero optimizer's gradients
       outputs = model(inputs)   # predict output
       loss = loss_fn(outputs, labels)   # calculate loss
       loss.backward()   # calculate loss gradient
       optimizer.step()   # adjust model parameters

The algorithm and all hyperparameters of the CoRe optimizer are explained in the file ``docs/documentation.pdf``.

License and Copyright Information
---------------------------------

The module core_optimizer is distributed under the BSD 3-Clause "New" or "Revised" License.
For more license and copyright information, see the file ``LICENSE.txt``.

How to Cite
-----------

When publishing results obtained with the CoRe optimizer, please cite
M. Eckhoff, M. Reiher, `Lifelong Machine Learning Potentials
<https://doi.org/10.1021/acs.jctc.3c00279>`_, J. Chem. Theory Comput. 2023, 19, 3509-3525
and
M. Eckhoff, M. Reiher, `CoRe optimizer: an all-in-one solution for machine learning
<https://doi.org/10.1088/2632-2153/ad1f76>`_, Mach. Learn.: Sci. Technol. 2024, 5, 015018.

Support and Contact
-------------------

In case you encounter any problems or bugs, please write a message to lifelong_ml@phys.chem.ethz.ch.
