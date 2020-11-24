# Notebooks

At the top level, in this directory, these notebooks demonstrate functionality, how to use aspects of the Demograhpic 
Modeling Module, and are well annotated with comments and markdown blocks discussing steps in each notebook.

* [01-get-geographies](./01-get-geographies.ipynb) - Frequently analysis is performed using standard geographies, areas 
already delineated by an administering agencies. Being able to easily discover, select and retrieve these geographies
is briefly demonstrated in this notebook.

* [02a-enrich-standard-geographies](./02a-enrich-standard-geographies.ipynb) - Once an area of interest is delineated, 
either by retrieving standard geographies or through other means, the process of enrichment enables retrieving scalar 
factors describing who people are living in delineated geographic areas. This process involves discovering which 
variables or factors are available, selecting the factors to use, and enriching the geographic areas.

* [02b-enrich-geographies](./02b-enrich-geographies.ipynb) - Many times the geographies to be enriched are _not_ 
standard geographies. They are created or determined by other means, and for analysis, demographic factors need to be 
apportioned to these geographies.

* [03-business-search](./03-business-search.ipynb) - An essential part of modeling and forecasting in the retail 
landscape is _getting_ features in the retail landscape, notably business locations. While locations are usually
known for an organization's brand, retrieving competitive and complimentary brand locations is also essential for
modeling the retail landscape.

Inside the `stash` directory are notebooks with ideas where things are being experimented with, and may or (more often) 
_may not_ actually work.