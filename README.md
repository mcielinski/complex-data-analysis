# complex-data-analysis
Assignments for the Complex Data Analysis course.


1. Collective classification

	* The chosen collective classification algorithm - ICA (Iterative Classification Algorithm) was implemented for the task of classification of labels of groups of employees. Subsequently, research was conducted on the selection of starting nodes for label discovery (selection method, percentage of discovered labels and type of internal classifier). In addition, a random algorithm was implemented for an analogous task and the obtained results were compared. For the set of experiments, indications of the metrics of accuracy, precision, sensitivity and F1 measure were recorded, the last of which was mainly used to measure the performance of the classifier and to compare the results.


2. Hierarchical classification

	* Familiarization with hierarchical classification methods (Flat Classification, Big-Bang, LCPN (Local classifier per parent node ), LCN (Local classifier per node), LCL (Local classifier per level)) for a selected dataset from [LINK](https://sites.google.com/site/hrsvmproject/datasets-hier).
	* Chosen dataset: imclef07a


3. Streaming data - concept drift detection

	* A mechanism was created that treats incoming instances as a stream of data, that is, without the ability to remember the entire string. Then, a mechanism was developed to detect concept drift, or more precisely, to suggest that new instances begin to come from a different distribution and inform the user about this fact (with additional information at which element it was detected).
	* Used concept drift detection methods (descriptions from [scikit-multiflow docs](https://scikit-multiflow.readthedocs.io/en/stable/api/api.html)): ADWIN, DDM, EDDM, HDDM_A, HDDM_W, KSWIN, PageHinkley