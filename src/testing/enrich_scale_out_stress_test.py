import math
import os.path
from pathlib import Path
import sys

import arcpy

sys.path.insert(0, os.path.abspath('../'))
from dm import Country
from dm._registry import get_ba_key_value

# DMA to test against
test_dma_name = 'seattle'

# get paths to data for stress testing
ba_data_dir = Path(get_ba_key_value('DataInstallDir', 'USA'))
gdb = ba_data_dir/'Data'/'Demographic Data'/'USA_ACS_2019.gdb'
bg_fc = gdb/'BlockGroups_bg'
dma_fc = gdb/'DMAs_dm'

# create a couple of layers for working with
dma_lyr = arcpy.management.MakeFeatureLayer(str(dma_fc), where_clause=f"UPPER(NAME) LIKE UPPER('%{test_dma_name}%')")
assert int(arcpy.management.GetCount(dma_lyr)[0]) == 1
bg_lyr = arcpy.management.MakeFeatureLayer(str(bg_fc))

# select only the block groups in the DMA
arcpy.management.SelectLayerByLocation(
    in_layer=bg_lyr,
    overlap_type='HAVE_THEIR_CENTER_IN',
    select_features=dma_lyr
)

# get the enrichment variables
enrich_vars = Country('USA').enrich_variables['enrich_str']

# starting variable count
n_vars = 20
interval = n_vars
total_vars = len(enrich_vars)
total_success = False


def enrich(n):
    """Attempt to enrich with a count of variables."""
    fail_cnt = 0
    success = False

    # if it fails, try a few times just to make sure
    while fail_cnt < 5:

        # get some random variables and see what happens
        try:
            var_lst = enrich_vars.sample(n, random_state=42)
            var_str = ';'.join(var_lst)
            arcpy.ba.EnrichLayer(bg_lyr,
                                 os.path.join(arcpy.env.scratchGDB, f'test_enrich_{len(var_lst):04d}_{fail_cnt}'),
                                 var_str)
            success = True
            break
        except Exception as e:
            fail_cnt += 1

    return success


# keep going until either the maximum number of variables is reached, the number of variables drops to one, or the
# interval drops to one, and we have found the enrichment threshold on this machine
while not total_success and n_vars > 1 and interval > 1:

    # try to enrich and store the result
    res = enrich(n_vars)

    # if enrichment was successful, double the interval
    if res is True:
        print(f'success: {n_vars}')
        interval *= 2
        n_vars = n_vars + interval

    # if not successful, divide the interval in half, and adjust the number of variables
    else:
        print(f'fail: {n_vars}')
        interval = math.floor(interval / 2)
        n_vars = n_vars - interval

    # if, when successful, the increased count is greater than the total number of variables, call it a win
    if n_vars > total_vars:
        total_success = True
