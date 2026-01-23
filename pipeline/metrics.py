"""
Evaluation metrics for text generation.

Computes BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, and ROUGE-L scores.
"""

from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU-1/2/3/4, METEOR, and ROUGE-L scores.
    
    Args:
        predictions: List of predicted text
        references: List of reference (ground truth) text
        
    Returns:
        Dictionary of metric scores
    """
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge_scores = []
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction()
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        
        # BLEU scores (n-gram precision)
        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), 
                             smoothing_function=smoothing.method1)
        bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0),
                             smoothing_function=smoothing.method1)
        bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0),
                             smoothing_function=smoothing.method1)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smoothing.method1)
        
        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)
        
        # METEOR score (considers synonyms and stemming)
        meteor = meteor_score([ref_tokens], pred_tokens)
        meteor_scores.append(meteor)
        
        # ROUGE-L score (longest common subsequence)
        rouge = scorer.score(ref, pred)
        rouge_scores.append(rouge['rougeL'].fmeasure)
    
    # Average all scores
    metrics = {
        'bleu1': sum(bleu1_scores) / len(bleu1_scores),     
        'bleu2': sum(bleu2_scores) / len(bleu2_scores),
        'bleu3': sum(bleu3_scores) / len(bleu3_scores),
        'bleu4': sum(bleu4_scores) / len(bleu4_scores),
        'meteor': sum(meteor_scores) / len(meteor_scores),
        'rouge_l': sum(rouge_scores) / len(rouge_scores),
    }
    
    return metrics
