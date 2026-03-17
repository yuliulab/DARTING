
# hotspot_finder

This module is used to identify activity hotspots for downstream molecular generation.
Here, we take MEK1 and MTOR as examples.

## Notebook
`get_hotspot_for_MEK1_MTOR.ipynb`


## Input
The input file `deal_bindingdb_alldata.csv` is a processed file derived from the original BindingDB dataset.  
It is not distributed with this repository. 
Users should download the BindingDB data separately and generate this file through their own preprocessing pipeline.
The input table is expected to contain at least the following columns:
* UniProt (SwissProt) Primary ID of Target Chain
* UniProt (SwissProt) Recommended Name of Target Chain
* region
* density
* IC50_dealed2  
In addition, the notebook uses a target mapping dictionary (targets_dict) to specify the target UniProt IDs and their simplified names.

## Analysis workflow
For each target, the notebook performs the following steps:
* Extract target-specific records from the processed BindingDB-derived dataset.
* Group compounds by region (R1, R2, R3).
* Select compounds located in the top-density fraction of each region.
* Compare IC50 values across regions.
* Estimate the median and mean IC50 values with 95% bootstrap confidence intervals.
* Perform pairwise Kolmogorov-Smirnov (KS) tests between regions and apply FDR correction.
* Select the optimal hotspot region based on:
* the lowest median IC50;
* if tied, the lowest mean IC50.
* Export hotspot-associated compounds for downstream molecular generation.

## Parameters

The current notebook uses:
* Output directory:
./test_data/step2_compare_ic_bw_region_v3
* Input file:
./deal_bindingdb_alldata.csv
* Top-density ratio:
top_ratio = 0.45
This means that, for each region, the notebook keeps the top 45% highest-density samples for IC50 comparison.

## Output
Results are saved under:

```python
./test_data/step2_compare_ic_bw_region_v3
```

## Usage
Open and run all cells in `get_hotspot_for_MEK1_MTOR.ipynb` in order.
