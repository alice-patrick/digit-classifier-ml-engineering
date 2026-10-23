# How to present this project in interviews

**Important:** This repository was prepared as an end-to-end starter project. Review the code and reproduce the results yourself before saying you independently developed or designed it. Be ready to describe every trade-off below.

## One-minute project summary

"I built a handwritten-digit classification prototype to demonstrate the complete ML workflow, not just model training. I compared logistic regression, random forest and RBF SVM on a validation set, selected the model using macro-F1 and evaluated it on a held-out test set. I also investigated which handwritten digits it misclassified, then exposed the selected model through a FastAPI endpoint and a small web demo."

## Defensible CV bullets (after you have reproduced and understood it)

- Compared Logistic Regression, Random Forest and RBF SVM on 1,797 handwritten-digit images; selected the model using a separate validation split.
- Achieved 98.1% accuracy and 0.980 macro-F1 on 360 held-out test images; documented 7 classification errors through a confusion matrix and a misclassification gallery.
- Built a FastAPI inference endpoint, input validation, eight automated tests and a Docker configuration for the reproducible prototype.

## Important technical questions

**Why macro-F1?** It treats each class equally, regardless of class frequency; that is a reasonable model-selection metric for multi-class digits. Accuracy is also shown for clarity.

**What prevents leakage?** Scaling is inside scikit-learn pipelines and fitted only on the training split. The test set never influences model selection. The exact training and validation indices are disjoint from the test indices.

**What is the biggest weakness?** The image size and sample are much simpler than real-world handwriting, there is only one evaluation split, and independently calibrated confidence has not been established.

**Why not use deep learning?** This deliberately scoped project demonstrates conventional ML, disciplined evaluation and packaging. A separate CNN comparison would be a future project.

**Is this production-ready?** No. It is a locally runnable, tested prototype. No authentication, rate limiting, monitoring, writer-independent evaluation or production readiness claims are made.

**What is the difference from the MSc fraud project?** The MSc prototype focuses on decision-making with limited investigation capacity and model-based risk scores. This project focuses on multi-class model benchmarking, visual error analysis, and serving predictions for a distinct image domain.
