# Notebooks

At the top level, in this directory, these notebooks demonstrate functionality, how to use aspects of the Demograhpic 
Modeling Module, and are well annotated with comments and markdown blocks discussing steps in each notebook.

* [01-working-with-countries](./01-working-with_countries) - The `Country` object is the object used to perform almost
all tasks using this modeling module. Introspection for discovering what country data is available is built into this
package. Hence, this is a good place to start to understand how to get up and running. It's a very short notebook, so
this one won't take much time to understand.

* [02-get-geography_levels](./01-get-geographies.ipynb) - Frequently analysis is performed using standard geography_levels, areas 
already delineated by an administering agencies. Being able to easily discover, select and retrieve these geography_levels
is briefly demonstrated in this notebook.

* [03a-enrich-standard-geography_levels](./02a-enrich-standard-geographies.ipynb) - Once an area of interest is delineated, 
either by retrieving standard geography_levels or through other means, the process of enrichment enables retrieving scalar 
factors describing who people are living in delineated geographic areas. This process involves discovering which 
variables or factors are available, selecting the factors to use, and enriching the geographic areas.

* [03b-enrich-geography_levels](./02b-enrich-geographies.ipynb) - Many times the geography_levels to be enriched are _not_ 
standard geography_levels. They are created or determined by other means, and for analysis, demographic factors need to be 
apportioned to these geography_levels.

* [04-business-search](./03-business-search.ipynb) - An essential part of modeling and forecasting in the retail 
landscape is _getting_ features in the retail landscape, notably business locations. While locations are usually
known for an organization's brand, retrieving competitive and complimentary brand locations is also essential for
modeling the retail landscape.

Inside the `stash` directory are notebooks with ideas where things are being experimented with, and may or (more often) 
_may not_ actually work.