# demographic-modeling-module

Demographic Modeling is _opinionated_ tooling for creating geographic factors for machine learning.

## Opinionated

No, this set of tooling written in Python is not going to have a political debate with you. Rather, while flexible 
enough to be used in a variety of ways, this tooling provides a clear way to perform analysis. This enables you to
get started and be productive as quickly as possible.

### A Few Opinions

#### DataFrames

There is a reason why DataFrames are so ubiqutous across all data science data tools. DataFrames, tabular data with
built in functionality, they make working with data. In this project, almost everything takes input data as a Pandas
DataFrame, and if the data is geographic, spatial, it is an Esri Spatially Enabled DataFrame.

#### Spatial Reference - WGS 84

Frequently one of the most difficult things about working with spatial data is knowing where it is. Spatial reference,
many times referred to as the the *projection*, is how we know where things are located on the face of the planet.
The most commonly used spatial reference is WGS84, the longitude and latitude used by GPS systems, including those in
smart phones. However, for those of us who spend our lives working with geographic data, how we know where something
is, or is *not*, is not as simple as it seems. Data can come in a variety of *spatial references*, and to make working
with data coming together potentially in differing spatial references easier, this package converts (*projects*) data
into WGS84 by default so the data is easier to deal with and the coordinates are easily recognized by most people,
geographers and data scientists alike.

## Getting Started

From the project directory, create an environment with all dependencies installed and linked.

```
> make env
```

If you want to manually activate the environment, this can be accomplished with the commmand.

```
> conda activate demographic-modeling
```

An example workflow can be found in the Jupyter notebooks in the `./notebooks` directory of the project. If you want 
to get started quickly, you can use the following command. It activates the environment and starts Jupyter Lab in
one consolidated step for you.

```
> make jupyter
```

<p><small>Project based on the <a target="_blank" href="https://github.com/knu2xs/cookiecutter-geoai">cookiecutter 
GeoAI project template</a>. This template, in turn, is simply an extension and light modification of the 
<a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science 
project template</a>. #cookiecutterdatascience</small></p>
