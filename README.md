# FeatureSelection
Feature Selection for improving predictive power


### Dependencies
* Runs on Python3
* Install all dependencies.
`pip3 install scikit-learn mlxtend joblib pandas numpy imbalanced-learn`

### Quickstart
* Place original data in `data/orig/`
* Run `1_prep_data.py`
* Explore which feature selection method is required in `2_feat_selection.py`

### Rebalancing data for imblanced samples
* In `3_rebalance_rolled_data`, change `data_src` to directory which has train and test data splits
* Mention method and target directory
```python
	bal = DataBalancer(data_src=`data/windowed/`,dst_dir=`data/windowed_balanced/`,seed=42)
	bal.rebalance_data(split=`train`,method=`smotetomek`)
	bal.rebalance_data(split=`test`,method=`none`)
```
* Methods
	* No oversampling : `none`,
	* Undersampling : `cluster` or `near_miss`
	* Oversampling : `smote` or `adasyn`
	* Combined : `smoteenn` or `smotetomek`
