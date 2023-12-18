# Radiotranscriptomics of non-small cell lung carcinoma for assessing high-level clinical outcomes using a machine learning-derived multi-modal signature 

Background: Multi-omics research has the potential to holistically capture intra-tumor variability, thereby improving therapeutic decisions by incorporating the key principles of precision medicine. The purpose of this study is to identify the most optimal method of integrating features from different sources, such as imaging, transcriptomics, and clinical data, to predict the survival and therapy response of non-small cell lung cancer patients. 

Methods: Radiomics and deep features were extracted from the volume of interest in pre-treatment CT examinations and then combined with RNA-seq and clinical data. Several machine learning classifiers were used to perform survival analysis and assess the patient’s response to adjuvant chemotherapy. The proposed analysis was evaluated on an unseen testing set in a k-fold cross-validation scheme. 

Results: The examined multi-omics model improved the AUC up to 0.10 on the unseen testing set (0.74±0.06) and the balance between sensitivity and specificity for predicting therapy response, resulting in less biased models and improving upon the either highly sensitive or highly specific single-source models. Accordingly, the survival analysis was also enhanced up to 0.20 by the proposed methodology in terms of the C-index (0.79±0.03).

Conclusion: Compared to single-source models, multi-omics integration significantly improved prediction performance, increased model stability, and decreased bias for both treatment response and survival analysis.

Published: BMC BioMedical Engineering OnLine
Link: https://doi.org/10.1186/s12938-023-01190-z

Lasso-based pipeline implementation:
https://github.com/NikiKou/deep_radiotranscriptomics_survival_analysis
