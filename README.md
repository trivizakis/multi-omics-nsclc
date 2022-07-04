# Deep Radiotranscriptomics of Non-Small Cell Lung Carcinoma for Assessing High-Level Clinical Outcomes using Multi-View Analysis

Background: Multi-omics research has the potential to holistically capture intra-tumor variability, thereby improving therapeutic decisions by incorporating the key principles of precision medicine. The purpose of this study is to use information from different sources, such as imaging, transcriptomics, and clinical data, to predict the survival and therapy response of non-small cell lung cancer patients.

Methods: Radiomics and deep features were extracted from the volume of interest in the studied CT examinations prior to treatment and combined with the RNA-seq and clinical data. A multi-view analysis was adapted for this multi-omics study, featuring SMOTE for balancing the class distributions, a combination of univariate and multivariate for feature selection, and an early feature integration technique. Several machine learning classifiers were used to perform the survival analysis and assess the patient’s response to neoadjuvant treatment. The analyses were evaluated on the unseen testing set in a k-fold cross-validation scheme.

Results: The examined multi-omics model improved the testing AUC by 0.04 and the balance between sensitivity and specificity for neoadjuvant treatment response, resulting in less biased models and improving upon the either highly sensitive or highly specific single-source models. Accordingly, the survival analysis was also enhanced from the proposed methodology by up to 0.22 in terms of C-index.

Published: Pending

Lasso-based pipeline implementation:
https://github.com/NikiKou/deep_radiotranscriptomics_survival_analysis
