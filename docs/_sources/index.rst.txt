.. demographic-modeling-module documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Demographic-Modeling-Module's documentation!
==========================================================

This project for Demographic Modeling does not introduce *new* functionality to ArcGIS,
but *does* make demographic modeling using machine learning *dramatically* easier. This
really is a much better organized evolution of the tools I developed to perform this
type of analysis for my own work with customers. Hence, my pain is your gain.

To get started with this project quickly, I suggest cloning the repo, install the package, and diving into the
examples in the Jupyter Notebooks included in the repo. Play with the notebook examples. Change them for your data, and
see what you can discover. From there, a lot more is included in the module documentation here to explore.

Quickstart
==========================================================

I can tell you're impatient because you are human, so here is how you get started quickly. First, ensure you have the
requirements, and then use these commands to start playing around on your own system.

Requirements
----------------------------------------------------------

* `ArcGIS Pro <https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview>`_
* `Business Analyst <https://www.esri.com/en-us/arcgis/products/arcgis-business-analyst/applications/desktop>`_ (Extends ArcGIS Pro)
* `Locally Installed Data Bundle <https://doc.arcgis.com/en/esri-demographics/data/us-intro.htm>`_ (most examples use the United States, but *should* work for any country)
* `Git <https://git-scm.com/download/win>`_ (for cloning the code repo)

Commands
-----------------------------------------------------------

First, clone the repo, and step into the project directory.

.. code-block::

   git clone https://github.com/knu2xs/demographic-modeling-module

   cd ./demographic-modeling-module

Next, take advantage of the make file to create a new Conda environment for you, and install
the source code for you using PIP.

.. code-block::

   make env

Now, jump into the notebooks directory and start looking around.

.. code-block::

   cd ./notebooks

   jupyter lab

Finally, once you've broken a few things, or want to figure out more, come back here and look
at the documentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation and Configuration<install>
   Demographic Modeling<dm>
   dm.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
