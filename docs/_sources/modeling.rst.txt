ModelingAccessor (``mdl``)
----------------------------------------------------------

The ModelingAccessor object (``df.mdl``), a Pandas DataFrame Accessor, likely is
going to be one of the most often used objects in this package. The
ModelingAccessor object is rarely, if ever, created directly. Rather,
it is accessed as a property of a Spatially Enabled DataFrame.

.. code-block:: python

    from dm import Country

    brand_name = 'ace hardware'

    # start by creating a country object instance
    usa = Country('USA')

    # get a geography to work with from locally installed data
    aoi_df = usa.cbsas.get('seattle')

    # use the DemographicModeling accessor to get block groups in the AOI
    bg_df = aoi_df.block_groups.get()

    # get the brand business locations
    biz_df = aoi_df.mdl.business.get_by_name(brand_name)

    # get the competition locations
    comp_df = aoi_df.mdl.business.get_competition(biz_df, local_threshold=3)

    # get current year key variables for enrichment
    e_vars = cntry.enrich_variables
    key_vars = e_vars[
        (e_vars.data_collection.str.startswith('Key'))
        & (e_vars.name.str.endswith('CY'))
    ]

    # use the DemographicModeling accessor to now enrich the block groups
    enrich_df = bg_df.mdl.enrich(key_vars)

    # get the drive distance and drive time to nearest three brand store locations for each block group
    bg_near_biz_df = enrich_df.mdl.proximity.get_nearest(biz_df, origin_id_column='ID', near_prefix='brand'))

    # now, do the same for competitor locations
    bg_near_biz_comp_df = bg_near_biz_df.mdl.proximity.get_nearest(
        origin_id_column='ID',
        near_prefix='comp',
        destination_count=6
        destination_columns_to_keep=['brand_name', 'brand_name_category']
    )

.. autoclass:: modeling.ModelingAccessor
   :members:

Business
----------------------------------------------------------

.. autoclass:: modeling.Business
   :members:

Proximity
----------------------------------------------------------

.. autoclass:: modeling.Proximity
   :members:

Country
----------------------------------------------------------

The country object is the foundational building block for
working with demographic data. This is due to data collection,
aggregation and dissemination methods used in Business Analyst.
Succinctly, this is how the data is organized.

.. autoclass:: modeling.Country
   :members:



