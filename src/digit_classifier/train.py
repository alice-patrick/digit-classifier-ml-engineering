"""Fit candidate classifiers, select on validation F1, evaluate once on test.

Usage: python -m digit_classifier.train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score,
    ConfusionMatrixDisplay,
)
from .modeling import build_candidates, load_splits, SEED

ROOT = Path(__file__).resolve().parents[2]


def metric_set(y, pred, probs) -> dict:
    return {
        'accuracy': round(float(accuracy_score(y, pred)), 4),
        'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
        'macro_precision': round(float(precision_score(y, pred, average='macro', zero_division=0)), 4),
        'macro_recall': round(float(recall_score(y, pred, average='macro', zero_division=0)), 4),
        'log_loss': round(float(log_loss(y, probs, labels=list(range(10)))), 4),
    }


def main(output: Path = ROOT) -> dict:
    output = Path(output)
    (output / 'reports' / 'figures').mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir(parents=True, exist_ok=True)
    d = load_splits()
    candidates = build_candidates()
    results = {}
    for name, model in candidates.items():
        model.fit(d.x_train, d.y_train)
        vpred = model.predict(d.x_validation)
        vprob = model.predict_proba(d.x_validation)
        results[name] = {'validation': metric_set(d.y_validation, vpred, vprob), 'model': model}

    # Tie break is deterministic. SVM is not assumed to win a priori.
    winner = sorted(results, key=lambda name: (-results[name]['validation']['macro_f1'], name))[0]
    best = results[winner]['model']
    # NOTE: This is the validation-selected fitted model trained on TRAIN only.
    # We do not re-fit on validation, retaining a clean held-out comparison.
    pred = best.predict(d.x_test)
    proba = best.predict_proba(d.x_test)
    test_metrics = metric_set(d.y_test, pred, proba)
    mis = np.flatnonzero(pred != d.y_test)
    per_class = classification_report(d.y_test, pred, labels=list(range(10)), output_dict=True, zero_division=0)
    selected_examples = []
    for i in mis[np.argsort(-np.max(proba[mis], axis=1))[:12]]:
        selected_examples.append({
            'test_index': int(i), 'dataset_index': int(d.ids_test[i]),
            'actual': int(d.y_test[i]), 'predicted': int(pred[i]),
            'confidence': round(float(proba[i].max()), 4),
        })
    result = {
        'dataset': 'scikit-learn digits (1,797 real 8x8 images)',
        'seed': SEED,
        'split': {'train':len(d.y_train), 'validation':len(d.y_validation), 'test':len(d.y_test)},
        'selection_metric': 'validation macro-F1',
        'candidate_validation_metrics': {n:x['validation'] for n,x in results.items()},
        'selected_model': winner,
        'test_metrics': test_metrics,
        'misclassified_count': int(len(mis)),
        'per_class': {k:{metric:round(float(v[metric]), 4) for metric in ['precision','recall','f1-score','support']}
                     for k,v in per_class.items() if k in {str(i) for i in range(10)}},
        'high_confidence_mistakes': selected_examples,
        'scope': 'A single stratified holdout. Results are illustrative, not production validation.',
    }
    (output / 'reports' / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    joblib.dump({'estimator':best, 'name':winner, 'test_metrics':test_metrics}, output / 'models' / 'model.joblib')

    names=list(results)
    vals=[results[n]['validation']['macro_f1'] for n in names]
    fig,ax=plt.subplots(figsize=(8,3.7))
    ax.barh(names, vals)
    ax.set_xlim(0,1.0)
    ax.set_xlabel('Validation macro-F1')
    ax.set_title('Model comparison — selected on validation data')
    for i,v in enumerate(vals):ax.text(min(v+0.008,.975),i,f'{v:.3f}',va='center')
    fig.tight_layout()
    fig.savefig(output/'reports'/'figures'/'model_comparison.png',dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5.7))
    cm=confusion_matrix(d.y_test,pred,labels=list(range(10)))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(ax=ax,values_format='d',cmap='Blues',colorbar=False)
    ax.set_title('Held-out test set — confusion matrix')
    fig.tight_layout()
    fig.savefig(output/'reports'/'figures'/'confusion_matrix.png',dpi=150)
    plt.close(fig)

    fig,axes=plt.subplots(3,4,figsize=(9.5,7.4))
    for ax,i in zip(axes.ravel(), mis[np.argsort(-proba[mis].max(axis=1))[:12]]):
        ax.imshow(d.x_test[i].reshape(8,8), cmap='gray_r', vmin=0, vmax=16, interpolation='nearest')
        ax.set_title(f'True {d.y_test[i]} → Pred {pred[i]} ({proba[i].max():.0%})',fontsize=9)
        ax.axis('off')
    for ax in axes.ravel()[min(12,len(mis)):] :ax.axis('off')
    fig.suptitle('Highest-confidence errors (held-out test)',fontsize=14)
    fig.tight_layout()
    fig.savefig(output/'reports'/'figures'/'error_examples.png',dpi=150)
    plt.close(fig)

    lines=[
        '# Error analysis — held-out test set','',
        '**Evaluation design:** one stratified 60/20/20 train/validation/test split, seed 42. '
        'Model choice uses validation macro-F1; the test set is evaluated only after selection.','',
        f"**Selected model:** {winner}",
        f"**Test performance:** accuracy {test_metrics['accuracy']:.3f}; macro-F1 {test_metrics['macro_f1']:.3f}; "
        f"macro-precision {test_metrics['macro_precision']:.3f}; macro-recall {test_metrics['macro_recall']:.3f}.",
        f"**Errors:** {len(mis)} of {len(d.y_test)} test images ({len(mis)/len(d.y_test):.1%}).",'',
        '## What the model gets wrong','',
        'The confusion matrix shows which digit pairs are mistaken for one another. '
        'The error gallery highlights the most *confident* incorrect predictions, '
        'which can be more concerning than uncertain mistakes.','',
        '![Confusion matrix](figures/confusion_matrix.png)','',
        '![Example errors](figures/error_examples.png)','',
        '## Per-digit recall','',
        '| Actual digit | Recall | Support |','|---:|---:|---:|'
    ]
    for k,v in result['per_class'].items(): lines.append(f"| {k} | {v['recall']:.1%} | {int(v['support'])} |")
    errors=sorted(((k,v['recall'],int(v['support'])) for k,v in result['per_class'].items()),key=lambda z:z[1])
    lines+=['',f"**Lowest recall in this split:** digit {errors[0][0]} ({errors[0][1]:.1%}; n={errors[0][2]}).",'',
        '## Limitations','',
        '- The dataset is small (8×8 digit images), unlike modern high-resolution handwritten images.',
        '- A single held-out split does not quantify variance across splits, writers, or deployment conditions.',
        '- Confidence scores are *not* independently calibrated or stress-tested.',
        '- No fairness or robustness claims; no external production deployment.','',
        '## Next experimental steps','',
        'Evaluate multiple splits or nested CV; test image noise/rotation; collect out-of-distribution examples; '
        'assess calibration and confidence-based abstention.'
    ]
    (output/'reports'/'error_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'selected':winner,'validation':results[winner]['validation'],
                      'test':test_metrics,'errors':len(mis)},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Reproduce digits experiment.')
    p.add_argument('--output', type=Path, default=ROOT)
    args=p.parse_args()
    main(args.output)
