.. demographic-modeling documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to documentation for Demographic-Modeling!
==========================================================

Demographic Modeling does not introduce *new* functionality to ArcGIS, but *does* make using ArcGIS Business Analyst
with machine learning *dramatically* easier. Demographic Modeling provides a single Python interface for
using ArcGIS Business Analyst for feature creation as part of a data preparation pipeline.

To get started with this project quickly, I suggest cloning the repo, install the package, and diving into the
examples in the Jupyter Notebooks included in the repo. Play with the notebook examples. Change them for your data, and
see what you can discover. From there, a lot more is included in the module documentation here to explore.

Quickstart
==========================================================

I can tell you're impatient because you are human, so here is how you get started quickly. First, ensure you have the
requirements, and then use these commands to start playing around on your own system.

Requirements
----------------------------------------------------------

The requirements vary based on how you are accessing Esri Business Analyst, whether through ArcGIS Pro or using
Business Analyst Web. The latter, Business Analyst Web, can be either ArcGIS Enterprise with Business Analyst Server,
or it can be simply using Business Analyst Web as part of ArcGIS Online.

ArcGIS Pro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `ArcGIS Pro <https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview>`_
* `Business Analyst <https://www.esri.com/en-us/arcgis/products/arcgis-business-analyst/applications/desktop>`_ (Extends ArcGIS Pro)
* `Locally Installed Data Bundle <https://doc.arcgis.com/en/esri-demographics/data/us-intro.htm>`_ (most examples use the USA, but *should* work for any country)
* `Git <https://git-scm.com/download/win>`_ (for cloning the code repo)

Business Analyst Web
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Conda (`MiniConda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com>`_)
* Python > 3.5
* `Git <https://git-scm.com/download/win>`_ (for cloning the code repo)
* Credentials with Access to Business Analyst Web instance (`ArcGIS Enterprise with Business Analyst Server <https://www.esri.com/en-us/arcgis/products/arcgis-business-analyst/applications/enterprise>`_ or `Business Analyst Web as part of ArcGIS Online <https://www.esri.com/en-us/arcgis/products/arcgis-business-analyst/applications/web-mobile-apps>`_)

Commands
-----------------------------------------------------------

First, clone the repo, and step into the project directory.

.. code-block:: bash

   git clone https://github.com/knu2xs/demographic-modeling-module

   cd ./demographic-modeling-module

Next, take advantage of the make file to create a new Conda environment for you, and install
the source code for you using PIP.

.. code-block:: bash

   make env

Now, jump into the notebooks directory and start looking around.

.. code-block:: bash

   cd ./notebooks

   jupyter lab

Finally, once you've broken a few things, or want to figure out more, come back here and look
at the documentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation and Configuration<install>
   Demographic Modeling<modeling>
   modeling.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
