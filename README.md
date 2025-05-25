# BERT Sentiment Analysis on IMDB Dataset

A deep learning project that implements sentiment analysis on movie reviews using BERT (Bidirectional Encoder Representations from Transformers) and the IMDB movie review dataset.

## Project Overview

This project fine-tunes a pre-trained BERT model to classify movie reviews as either positive or negative sentiment. The model achieves high accuracy by leveraging BERT's contextual understanding of text and transfer learning capabilities.

## Features

- **BERT-based Architecture**: Uses pre-trained BERT model for superior text understanding
- **IMDB Dataset**: Trained on 50,000 labeled movie reviews (25k train, 25k test)
- **GPU Acceleration**: Optimized for CUDA-enabled training
- **Comprehensive Evaluation**: Detailed performance metrics and classification reports
- **Data Preprocessing**: Text cleaning and tokenization pipeline
- **Model Checkpointing**: Save and load trained models
- **Progress Tracking**: Real-time training progress with tqdm

## Dataset Information

- **Source**: IMDB Movie Review Dataset
- **Size**: 50,000 movie reviews
- **Classes**: Binary classification (Positive/Negative)
- **Split**: 25,000 training samples, 25,000 test samples
- **Additional**: 50,000 unlabeled reviews for unsupervised learning

## Requirements

Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Installation

1. Clone this repository:
```bash
git clone https://github.com/RAKSAurum/Sentiment-Analysis-Using-BERT.git
cd Sentiment-Analysis-Using-BERT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook Sentiment_Analysis_IMDB_Dataset_BERT.ipynb
```

## Usage

### Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Enable GPU runtime: Runtime → Change runtime type → GPU
3. Run all cells sequentially

### Local Environment

1. Ensure you have a CUDA-compatible GPU (optional but recommended)
2. Open and run the Jupyter notebook
3. The dataset will be automatically downloaded on first run

## Model Architecture

The project uses the following architecture:

- **Base Model**: BERT (bert-base-uncased)
- **Classification Head**: Linear layer for binary classification
- **Optimizer**: AdamW with learning rate scheduling
- **Loss Function**: CrossEntropyLoss
- **Regularization**: Dropout layers in BERT

## Key Components

### Data Processing
- Text preprocessing and cleaning
- BERT tokenization with padding and truncation
- Custom PyTorch Dataset class for efficient data loading

### Model Training
- Fine-tuning pre-trained BERT model
- Learning rate scheduling with ReduceLROnPlateau
- Gradient clipping for training stability
- Early stopping to prevent overfitting

### Evaluation
- Accuracy metrics
- Precision, Recall, and F1-score
- Confusion matrix analysis
- Classification report

## Performance

Expected model performance:
- **Accuracy**: ~92-95%
- **Training Time**: ~30-45 minutes on GPU
- **Memory Requirements**: 8GB+ GPU memory recommended

## Technical Details

### Hyperparameters
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Max Sequence Length**: 512
- **Epochs**: 3-5
- **Warmup Steps**: 500

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ for dataset and model files

## Model Training Process

1. **Data Preparation**: Tokenize and preprocess text data
2. **Model Setup**: Initialize BERT model with classification head
3. **Training Loop**: Fine-tune model with backpropagation
4. **Validation**: Evaluate performance on test set
5. **Model Saving**: Save trained model for future use

## Results Visualization

The notebook includes:
- Training/validation loss curves
- Accuracy progression over epochs
- Confusion matrix heatmap
- Sample predictions with confidence scores

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [IMDB Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

## Acknowledgments

- Google Research for the BERT model
- Hugging Face for the Transformers library
- Stanford AI Lab for the IMDB dataset
- PyTorch team for the deep learning framework
