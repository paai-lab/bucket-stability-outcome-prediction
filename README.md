# Assessing the quality of outcome-based business process predictive monitoring models
This repository handles the codes for the paper titled "Assessing the quality of outcome-based business process predictive monitoring models". The experiments use the approach proposed in "https://github.com/irhete/predictive-monitoring-benchmark" by Teinemaa et al. (2019). 
For trace bucketing, following techniques are used:
* Prefix-length bucketing
* Clustering bucketing

For sequence encoding, following technique is used:
* Index-based encoding

The experiment is conducted using three classifiers:
* Random forest (RF)
* Gradient Boosting Machine (GBM)
* Extreme Gradient Boosting (XGB)

Based on predicted probabilities and performance from the above classifiers, this paper uses stability_calculator.py to calculate three quality metrics as in below:
* Overall bucket performance (OBP)
* Intra-bucket prediction stability (IBS)
* Cross-bucket performance stability (XBS)

## References
Teinemaa, I., Dumas, M., Rosa, M. L., & Maggi, F. M. (2019). Outcome-oriented predictive process monitoring: Review and benchmark. ACM Transactions on Knowledge Discovery from data (TKDD), 13(2), 1-57
