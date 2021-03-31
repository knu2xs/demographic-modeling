Country
----------------------------------------------------------

The country object is the foundational building block for
working with demographic data. This is due to data collection,
aggregation and dissemination methods.

.. autoclass:: modeling.Country
   :members:

ModelingAccessor (`mdl`)
----------------------------------------------------------

Besides the Country object, the DemographicModeling object, a Pandas
DataFrame Accessor, likely is going to be one of the most often used
objects in this package. The ModelingAccessor object rarely, if ever,
is created directly. Rather, it is accessed as a property of a Spatially
Enabled DataFrame.

.. code-block:: python

    from dm import Country

    # start by creating a country object instance
    usa = Country('USA')

    # get a geography to work with from locally installed data
    aoi_df = usa.cbsas.get('seattle')

    # use the DemographicModeling accessor to get block groups in the AOI
    bg_df = aoi_df.block_groups.get()

    # get current year key variables for enrichment
    e_vars = cntry.enrich_variables
    key_vars = e_vars[
        (e_vars.data_collection.str.startswith('Key'))
        & (e_vars.name.str.endswith('CY'))
    ]

    # use the DemographicModeling accessor to now enrich the block groups
    enrich_df = ta_df.mdl.enrich(key_vars)

.. autoclass:: modeling.ModelingAccessor
   :members:

Business
----------------------------------------------------------

.. autoclass:: modeling.Business
   :members: