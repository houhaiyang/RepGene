**Table S1: Performance comparison between ESM-C and RepGene on protein single-modality tasks.**

| Task type         | Task name                                    | Sample size | ESM-C (Mean ± SD) | RepGene (Mean ± SD) |
|:------------------|:---------------------------------------------|------------:|---:|---:|
| Binary Tasks      | TF vs non-TF                                 |       2,765 | 0.860 ± 0.014 | **0.913** ± 0.012 |
| Binary Tasks      | long vs short range TF                       |         174 | 0.599 ± 0.118 | **0.640** ± 0.091 |
| Binary Tasks      | bivalent vs non-methylated                   |         133 | **0.880** ± 0.039 | 0.863 ± 0.043 |
| Binary Tasks      | dosage sensitive vs insensitive TF           |         487 | 0.888 ± 0.042 | **0.971** ± 0.007 |
| Binary Tasks      | CCD Transcript                               |       1,631 | 0.569 ± 0.022 | **0.665** ± 0.026 |
| Binary Tasks      | CCD Protein                                  |       1,429 | 0.532 ± 0.020 | **0.561** ± 0.027 |
| Binary Tasks      | N1 network                                   |       1,126 | 0.646 ± 0.043 | **0.750** ± 0.035 |
| Binary Tasks      | N1 targets                                   |         290 | 0.635 ± 0.077 | **0.725** ± 0.090 |
| Binary Tasks      | HLA class I vs class II                      |          44 | 0.235 ± 0.102 | **0.557** ± 0.221 |
| Categorical Tasks | Pathology prognostics - Breast cancer        |      17,445 | 0.556 ± 0.025 | **0.580** ± 0.038 |
| Categorical Tasks | Pathology prognostics - Cervical cancer      |      17,315 | **0.573** ± 0.016 | 0.573 ± 0.026 |
| Categorical Tasks | Pathology prognostics - Colorectal cancer    |      16,981 | **0.582** ± 0.021 | 0.560 ± 0.018 |
| Categorical Tasks | Pathology prognostics - Endometrial cancer   |      17,237 | 0.554 ± 0.018 | **0.555** ± 0.018 |
| Categorical Tasks | Pathology prognostics - Glioma               |      17,443 | 0.585 ± 0.014 | **0.605** ± 0.041 |
| Categorical Tasks | Pathology prognostics - Head and neck cancer |      17,437 | 0.535 ± 0.016 | **0.557** ± 0.021 |
| Categorical Tasks | Pathology prognostics - Liver cancer         |      16,723 | 0.659 ± 0.018 | **0.691** ± 0.005 |
| Categorical Tasks | Pathology prognostics - Lung cancer          |      17,607 | 0.525 ± 0.019 | **0.547** ± 0.005 |
| Categorical Tasks | Pathology prognostics - Melanoma             |      17,099 | 0.576 ± 0.047 | **0.640** ± 0.022 |
| Categorical Tasks | Pathology prognostics - Ovarian cancer       |      17,718 | 0.585 ± 0.005 | **0.601** ± 0.013 |
| Categorical Tasks | Pathology prognostics - Pancreatic cancer    |      17,602 | 0.562 ± 0.007 | **0.591** ± 0.026 |
| Categorical Tasks | Pathology prognostics - Prostate cancer      |      17,413 | 0.596 ± 0.051 | **0.616** ± 0.037 |
| Categorical Tasks | Pathology prognostics - Renal cancer         |      17,382 | 0.583 ± 0.009 | **0.597** ± 0.005 |
| Categorical Tasks | Pathology prognostics - Stomach cancer       |      18,009 | 0.539 ± 0.014 | **0.581** ± 0.052 |
| Categorical Tasks | Pathology prognostics - Testis cancer        |      18,021 | 0.573 ± 0.042 | **0.707** ± 0.069 |
| Categorical Tasks | Pathology prognostics - Thyroid cancer       |      17,206 | 0.541 ± 0.035 | **0.562** ± 0.031 |
| Categorical Tasks | Pathology prognostics - Urothelial cancer    |      17,277 | 0.563 ± 0.009 | **0.583** ± 0.018 |
| Categorical Tasks | RNA cancer distribution                      |      19,588 | 0.708 ± 0.004 | **0.767** ± 0.005 |
| Categorical Tasks | RNA cancer specificity                       |      19,588 | 0.712 ± 0.007 | **0.771** ± 0.002 |
| Categorical Tasks | Secretome function                           |       2,736 | 0.709 ± 0.008 | **0.819** ± 0.011 |
| Categorical Tasks | Secretome location                           |       2,767 | 0.697 ± 0.011 | **0.791** ± 0.012 |
| Categorical Tasks | Blood expression cluster                     |      12,697 | 0.586 ± 0.006 | **0.599** ± 0.007 |
| Categorical Tasks | Brain expression cluster                     |      17,590 | 0.617 ± 0.002 | **0.643** ± 0.001 |
| Categorical Tasks | Cell line expression cluster                 |      19,784 | 0.620 ± 0.005 | **0.665** ± 0.003 |
| Categorical Tasks | RNA blood cell distribution                  |      19,784 | 0.673 ± 0.006 | **0.731** ± 0.004 |
| Categorical Tasks | RNA blood cell specificity                   |      19,784 | 0.697 ± 0.007 | **0.763** ± 0.005 |
| Categorical Tasks | RNA blood lineage distribution               |      19,784 | 0.690 ± 0.005 | **0.748** ± 0.006 |
| Categorical Tasks | RNA blood lineage specificity                |      19,784 | 0.706 ± 0.003 | **0.773** ± 0.006 |
| Categorical Tasks | RNA brain regional distribution              |      19,784 | 0.708 ± 0.008 | **0.793** ± 0.007 |
| Categorical Tasks | RNA brain regional specificity               |      19,784 | 0.705 ± 0.008 | **0.782** ± 0.008 |
| Categorical Tasks | RNA cell line distribution                   |      19,784 | 0.720 ± 0.006 | **0.785** ± 0.004 |
| Categorical Tasks | RNA cell line specificity                    |      19,784 | 0.718 ± 0.005 | **0.784** ± 0.007 |
| Categorical Tasks | RNA mouse brain regional distribution        |      16,655 | 0.740 ± 0.014 | **0.791** ± 0.007 |
| Categorical Tasks | RNA mouse brain regional specificity         |      16,655 | 0.755 ± 0.007 | **0.802** ± 0.003 |
| Categorical Tasks | RNA pig brain regional distribution          |      16,595 | 0.726 ± 0.008 | **0.781** ± 0.008 |
| Categorical Tasks | RNA pig brain regional specificity           |      16,595 | 0.723 ± 0.011 | **0.782** ± 0.005 |
| Categorical Tasks | RNA single cell type distribution            |      19,761 | 0.699 ± 0.004 | **0.758** ± 0.004 |
| Categorical Tasks | RNA single cell type specificity             |      19,761 | 0.630 ± 0.003 | **0.682** ± 0.005 |
| Categorical Tasks | RNA tissue distribution                      |      19,784 | 0.705 ± 0.007 | **0.768** ± 0.004 |
| Categorical Tasks | RNA tissue specificity                       |      19,784 | 0.672 ± 0.004 | **0.724** ± 0.003 |
| Categorical Tasks | Single cell expression cluster               |      19,016 | 0.639 ± 0.003 | **0.686** ± 0.005 |
| Categorical Tasks | Tissue expression cluster                    |      18,355 | 0.656 ± 0.005 | **0.700** ± 0.004 |
| Multi Label Tasks | Molecular function                           |      10,991 | 0.511 ± 0.002 | **0.702** ± 0.007 |
| Multi Label Tasks | Protein class                                |      19,784 | 0.534 ± 0.000 | **0.689** ± 0.003 |
| Multi Label Tasks | Subcellular location                         |      13,039 | 0.539 ± 0.004 | **0.571** ± 0.006 |
| Multi Label Tasks | Biological process                           |      10,796 | 0.524 ± 0.003 | **0.691** ± 0.005 |
| Multi Label Tasks | Disease involvement                          |       5,837 | 0.511 ± 0.004 | **0.580** ± 0.008 |
| Multi Label Tasks | RNA tissue cell type enrichment              |      13,957 | 0.498 ± 0.001 | **0.526** ± 0.002 |
| Multi Label Tasks | UniProt keyword Ligand                       |       6,688 | 0.546 ± 0.002 | **0.702** ± 0.007 |
| Multi Label Tasks | UniProt keyword Domain                       |      13,474 | 0.640 ± 0.006 | **0.815** ± 0.005 |
| Multi Label Tasks | UniProt keyword PTM                          |      14,001 | 0.638 ± 0.004 | **0.743** ± 0.007 |

**Note:** This table presents the evaluation of gene representation models across diverse biological tasks . Higher average performance values are highlighted in **bold**. All tests were conducted using a logistic regression classifier with 5-fold cross-validation, strictly following the Gene Benchmark methodology described in the paper. Results are presented in a Mean ± SD format . Specifically, the mean ROC-AUC metric was utilized to evaluate binary classification tasks, while the weighted one-vs-rest mean ROC-AUC was applied for both categorical and multi-label classification tasks.
