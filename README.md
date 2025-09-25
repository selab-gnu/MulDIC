# MulDIC: A Multimodal Deep Learning Model for Cross-Project Issue Classification

[![IEEE Access](https://img.shields.io/badge/Published-IEEE%20Access-blue)](https://ieeexplore.ieee.org/document/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Research-green)](LICENSE)

MulDIC is the first multimodal deep learning model for cross-project, multi-class issue classification that leverages text, images, and code snippets from GitHub issue reports. This research addresses the limitations of traditional unimodal approaches by integrating heterogeneous information to achieve superior classification performance across diverse software projects.

## 🎯 Overview

Software issue classification is critical for effective project management and bug tracking. Traditional approaches rely solely on textual information, missing valuable context provided by images and code snippets in issue reports. MulDIC overcomes these limitations by:

- **Multimodal Learning**: Combining text descriptions, code snippets, and visual content
- **Cross-Project Classification**: Training on diverse projects to ensure generalizability
- **Multi-Class Support**: Classifying issues into 10 relevant categories instead of binary classification
- **State-of-the-Art Performance**: Achieving 5.50-7.01% improvement in F1-Score over unimodal baselines

## 📊 Research Contributions

### 1. First Multi-Class Multimodal Model
- Novel architecture combining text, image, and code modalities
- Classification into 10 relevant GitHub labels: bug, enhancement, question, feature, invalid, duplicate, wontfix, help-wanted, documentation, good-first-issue

### 2. Large-Scale Cross-Project Evaluation
- Evaluated across over 10,000 GitHub projects
- Dataset: 35,054 projects with 6,144,475 issue reports
- Multimodal subset: 10,913 projects with 81,450 issue reports containing text, images, and code

### 3. Comprehensive Performance Analysis
- Statistical significance testing with Mann-Whitney U-test
- Effect size analysis confirming practical significance
- Detailed ablation studies on modality contributions

## ⛏️ Dataset

### Zenedo
- Note (25 Sep): The dataset is currently being prepared for upload.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17194028.svg)](https://doi.org/10.5281/zenodo.17194028)

## 🏗️ Architecture

### Model Variants
- **MulDIC_TI**: Text + Image classification
- **MulDIC_TC**: Text + Code classification  
- **MulDIC_TIC**: Text + Image + Code classification (Full Model)
- **Text Only**: Baseline unimodal model

### CNN-Based Architecture
```
Input: (Text, Image, Code)
  ↓
Feature Extraction:
├── Text CNN Channel (TextCNN with 2,3,4,5-gram kernels)
├── Image CNN Channel (3x3 conv layers + max pooling)
└── Code CNN Channel (TextCNN with specialized preprocessing)
  ↓
Feature Integration: Element-wise Multiplication
  ↓
Classification: Fully Connected + Softmax → 10 Classes
```

## 📁 Dataset Structure

### Data Distribution by Label
| Label | Count | Percentage |
|-------|--------|------------|
| bug | 51,130 | 62.8% |
| enhancement | 8,273 | 10.2% |
| question | 7,285 | 8.9% |
| feature | 5,490 | 6.7% |
| invalid | 2,922 | 3.6% |
| duplicate | 2,489 | 3.1% |
| wontfix | 1,619 | 2.0% |
| help-wanted | 1,230 | 1.5% |
| documentation | 567 | 0.7% |
| good-first-issue | 445 | 0.5% |

### Directory Structure
```
multimodal_BothCodeImage_dataset/
├── train_data/              # 10-class training data
├── test_data/               # 10-class test data  
├── train_data_4Proj/        # 4-project specific training
├── test_data_4Proj/         # 4-project specific testing
└── IssueData_TotalProject_BothCodeImage_10labels.csv
```

## 🚀 Getting Started

### Data Preprocessing
The preprocessing pipeline includes:

1. **Text Preprocessing**:
   - Case-folding and tokenization
   - Stop-word removal using NLTK
   - Non-alphabetic token removal

2. **Code Preprocessing**:
   - Comment removal
   - Special token replacement ('\n', '\t')
   - Structure-preserving tokenization

3. **Image Preprocessing**:
   - Resize to 16×16 pixels
   - Normalization to [-1, 1] range

### Training Models

#### 1. Text+Code Classification (4 Projects)
```bash
python src/Main_MulDIC_4Proj_TC.py
```

#### 2. Text+Image Classification
```bash
python src/Main_MulDIC_TI.py
```

#### 3. Full Multimodal Classification
```bash
python src/Main_MulDIC_TIC.py
```

#### 4. Binary Classification
```bash
python src/Main_MulDIC_BinaryCLS_TIC.py
```

### Model Configuration
```python
# Key hyperparameters
EMBEDDING_SIZE = 300
MAX_LENGTH = 100
BATCH_SIZE = 256
EPOCHS = 1000
IMAGE_SIZE = (16, 16)
```

## 📈 Experimental Results

### Performance Comparison
| Model | Precision | Recall | F1-Score | Improvement |
|-------|-----------|---------|----------|-------------|
| Text Only | 70.29% | 63.39% | 65.19% | - (baseline) |
| MulDIC_TI | **72.20%** | **72.98%** | **72.20%** | +7.01% |
| MulDIC_TC | 71.05% | 71.75% | 70.96% | +5.77% |
| MulDIC_TIC | 70.83% | 71.14% | 70.69% | +5.50% |

### Statistical Significance
- All multimodal models show p-values < 0.05 (Mann-Whitney U-test)
- Effect sizes range from 0.20-0.25 (medium practical significance)
- Results demonstrate both statistical and practical significance

### Key Findings
1. **Text+Image combination most effective**: 7.01% F1-Score improvement
2. **Code data provides substantial benefit**: 5.77% F1-Score improvement  
3. **All multimodal approaches outperform baseline**: Consistent improvements across metrics
4. **Cross-project generalization**: Effective performance across diverse project types

## 🔧 Implementation Details

### Feature Extraction
- **Text/Code**: TextCNN with multiple kernel sizes (2,3,4,5-grams)
- **Image**: CNN with 3×3 convolution kernels and max-pooling
- **Fusion**: Element-wise multiplication for modality integration

### Model Architecture Components
```python
class MulDIC(nn.Module):
    def __init__(self):
        # Text/Code CNN channels
        self.text_channel = TextCNN(embedding_dim=300)
        self.code_channel = TextCNN(embedding_dim=300) 
        # Image CNN channel
        self.image_channel = ImageCNN(input_size=(16,16,3))
        # Fusion and classification layers
        self.fusion = ElementWiseMultiplication()
        self.classifier = nn.Linear(256, 10)
```

### Evaluation Metrics
- **Weighted Precision/Recall/F1-Score**: Accounts for class imbalance
- **Cross-Entropy Loss**: Standard multi-class classification loss
- **Statistical Testing**: Mann-Whitney U-test for significance

## 📊 Ablation Studies

### Research Questions Addressed
- **RQ1**: Does text+image improve classification? ✅ **+7.01% F1-Score**
- **RQ2**: Does text+code improve classification? ✅ **+5.77% F1-Score**  
- **RQ3**: Does text+image+code improve classification? ✅ **+5.50% F1-Score**

### Modality Contribution Analysis
- **Images**: Most effective additional modality
- **Code**: Significant performance boost over text-only
- **Combined**: Benefits from heterogeneous information sources


## 📝 Citation

If you use this work in your research, please cite our paper:

```bibtex

```

## 🔗 Links

[//]: # (- **Paper**: [IEEE Access Publication]&#40;https://ieeexplore.ieee.org/document/&#41;)

[//]: # (- **Dataset**: [GitHub Repository]&#40;https://github.com/selab-gnu/MulDIC&#41;)

[//]: # (- **Authors**: Gyeongsang National University Software Engineering Lab)

## 📧 Contact

For questions about the research, implementation, or dataset:

**First Author**: changwon kwak (chang_26@naver.com)

Department of AI Convergence Engineering  
Gyeongsang National University, South Korea

