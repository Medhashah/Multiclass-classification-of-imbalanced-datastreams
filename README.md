# Multiclass-classification-of-imbalanced-datastreams

The project proposes HECMI (Hybrid Ensemble technique for Classification of Multiclass Imbalanced data) to deal with the problem of multiclass imbalance data which has more than one majority and minority class.HECMI is generic in nature and can be used for any domain.It is a hybrid of data based and algorithm based method for imbalanced data.The base classifier of the ensemble is selected
from the existing traditional models with the best cross-validation score. The data is initially partitioned into n parts for n iterations.A data-based approach of oversampling minority class instances is used to balance the data in iterations.The class with least recall is the one misclassified the most and should be paid attention by the classifier.Instances of this class are oversampled and added to the next data part in training.Also, the instances of classes with recall less than threshold are added to the next data part.Final prediction is done by taking the majority of votes of the classifiers

publication link:https://link.springer.com/chapter/10.1007/978-981-13-3338-5_11
