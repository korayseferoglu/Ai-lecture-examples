------------------------------------------------------------
Breast Cancer Wisconsin (Diagnostic) Dataset - DATA README
------------------------------------------------------------

Source:
University of Wisconsin Hospitals, Madison – Dr. William H. Wolberg
Publication:
UCI Machine Learning Repository / scikit-learn (load_breast_cancer)

------------------------------------------------------------
1. GENERAL DESCRIPTION
------------------------------------------------------------
This dataset contains real clinical data used for breast cancer diagnosis based on 
cellular morphology measurements. Each row represents a breast tissue sample from a 
patient. Each column corresponds to a numerical feature extracted from the microscopic 
image of the cell nuclei.

Goal: Using 30 numerical features, predict whether the tumor is 
benign (non-cancerous) or malignant (cancerous).

------------------------------------------------------------
2. CLINICAL BACKGROUND
------------------------------------------------------------
Samples were obtained using fine needle aspiration (FNA) from breast masses.
The nuclei in these samples were digitized using computer-aided image analysis.

Malignant (cancerous) tumors:
- Typically have larger, irregular, and asymmetric nuclei.

Benign (non-cancerous) tumors:
- Have smaller, smooth, and symmetric nuclei.

------------------------------------------------------------
3. FEATURES (ATTRIBUTES)
------------------------------------------------------------
There are 30 numerical columns describing various geometric and textural 
characteristics of the cell nuclei.

Feature prefixes:
- mean ...  -> Mean value of the feature
- se ...    -> Standard error (variation)
- worst ... -> Maximum (worst-case) value

Examples of features:
- mean radius ........ Average size of the cell nuclei
- mean texture ....... Variation in pixel intensity (texture roughness)
- mean perimeter ..... Nucleus perimeter
- mean area .......... Nucleus area
- mean smoothness .... Surface smoothness
- mean symmetry ...... Symmetry of the nuclei
- mean fractal_dimension ... Fractal dimension (structural complexity)

------------------------------------------------------------
4. TARGET (LABEL)
------------------------------------------------------------
The 'target' column represents two diagnostic classes:
0 = malignant (cancerous)
1 = benign (non-cancerous)

------------------------------------------------------------
5. TECHNICAL DETAILS
------------------------------------------------------------
- Number of samples: 569
- Number of features: 30
- Number of classes: 2
- Data type: Numerical (float)
- Missing values: None
- Source: scikit-learn / UCI Machine Learning Repository

------------------------------------------------------------
6. APPLICATION AREAS
------------------------------------------------------------
- Machine learning: Binary classification
- Medical informatics: Computer-aided cancer diagnosis
- Feature importance and model evaluation studies
- ROC analysis and accuracy metrics

------------------------------------------------------------
7. REFERENCES
------------------------------------------------------------
W.N. Street, W.H. Wolberg, and O.L. Mangasarian.
"Nuclear feature extraction for breast tumor diagnosis."
IS&T/SPIE 1993 International Symposium on Electronic Imaging: 
Science and Technology, 1905:861–870, 1993.

scikit-learn dataset info:
https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset

------------------------------------------------------------
Prepared by: Enes Koray Seferoğlu
Description: This file summarizes the structure and meaning of the Breast Cancer dataset.
------------------------------------------------------------
