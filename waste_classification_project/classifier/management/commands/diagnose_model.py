from django.core.management.base import BaseCommand
import os
import sys
from collections import defaultdict, Counter

from classifier.utils import WasteClassifier


class Command(BaseCommand):
    help = 'Prints model information and optionally heuristically maps model output indices to class labels using a labeled sample directory.'

    def add_arguments(self, parser):
        parser.add_argument('--sample-dir', type=str, help='Path to a directory with subfolders for each true label containing sample images.')
        parser.add_argument('--max-samples-per-class', type=int, default=50, help='Limit samples per class to speed up mapping (default 50).')

    def handle(self, *args, **options):
        sample_dir = options.get('sample_dir')
        max_per = options.get('max_samples_per_class') or 50

        self.stdout.write('Initializing WasteClassifier (this may load the model)...')
        wc = WasteClassifier()

        # Basic info
        classes = wc.classes
        self.stdout.write(f'Known classes (len={len(classes)}): {classes}')

        model = wc.model
        if model is None:
            self.stdout.write(self.style.WARNING('Warning: model is not loaded (wc.model is None). Predictions will not work.'))
            return

        # Print model output shape and summary (small)
        try:
            out_shape = getattr(model, 'output_shape', None)
            self.stdout.write(f'Model output_shape: {out_shape}')
        except Exception:
            pass

        try:
            self.stdout.write('Model summary:')
            model.summary(print_fn=lambda s: self.stdout.write(s))
        except Exception:
            self.stdout.write(self.style.WARNING('Could not print full model summary.'))

        # If no sample dir provided, we're done
        if not sample_dir:
            self.stdout.write(self.style.SUCCESS('No sample directory given. Diagnosis complete.'))
            return

        # Validate sample dir
        if not os.path.exists(sample_dir) or not os.path.isdir(sample_dir):
            self.stdout.write(self.style.ERROR(f'Sample dir does not exist or is not a directory: {sample_dir}'))
            return

        # Gather labeled files: expect subfolders named by true label
        self.stdout.write(f'Collecting labeled samples from {sample_dir} (max {max_per} per class)...')
        labeled = []  # tuples (true_label, filepath)
        for label in sorted(os.listdir(sample_dir)):
            label_dir = os.path.join(sample_dir, label)
            if not os.path.isdir(label_dir):
                continue
            files = []
            for root, _, fnames in os.walk(label_dir):
                for fname in fnames:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        files.append(os.path.join(root, fname))
            files = files[:max_per]
            for p in files:
                labeled.append((label, p))

        if not labeled:
            self.stdout.write(self.style.ERROR('No labeled image files found in sample directory. Ensure subfolders per label with images inside.'))
            return

        self.stdout.write(f'Running predictions on {len(labeled)} samples...')
        # Map predicted index -> Counter(true_label)
        pred_to_true = defaultdict(Counter)
        true_counts = Counter()
        failed = 0

        for true_label, path in labeled:
            try:
                res = wc.predict(path)
                if not res:
                    failed += 1
                    continue
                # If predict returned label strings, capture index if possible
                pred_label = res.get('class')
                # If the predicted label is of form 'index_N', extract the index
                if isinstance(pred_label, str) and pred_label.startswith('index_'):
                    try:
                        idx = int(pred_label.split('_', 1)[1])
                    except Exception:
                        idx = None
                else:
                    # Map label back to index via wc.classes if possible
                    try:
                        idx = wc.classes.index(pred_label)
                    except Exception:
                        idx = None

                if idx is None:
                    # fallback: try to find highest-probability index in all_predictions
                    allp = res.get('all_predictions') or {}
                    if allp:
                        # find numeric-index key if present, else skip
                        numeric_keys = [k for k in allp.keys() if k.startswith('index_')]
                        if numeric_keys:
                            idx = int(sorted(numeric_keys)[0].split('_', 1)[1])
                if idx is None:
                    failed += 1
                    continue

                pred_to_true[idx][true_label] += 1
                true_counts[true_label] += 1
            except Exception as e:
                failed += 1

        self.stdout.write(f'Prediction run complete. Failed: {failed}.')

        # Summarize mapping
        self.stdout.write('\nHeuristic mapping from predicted index -> most common true label:')
        suggested = {}
        for idx, counter in sorted(pred_to_true.items()):
            most_common = counter.most_common()
            if not most_common:
                continue
            label, count = most_common[0]
            suggested[idx] = label
            self.stdout.write(f'  index {idx} -> {label} ({count} votes, details: {most_common})')

        # Print per-true-class confusion summary
        self.stdout.write('\nPer-true-class summary (how samples of each true label were distributed across predicted indices):')
        # invert pred_to_true to true_label -> Counter(idx->count)
        true_to_pred = defaultdict(Counter)
        for idx, counter in pred_to_true.items():
            for tlabel, cnt in counter.items():
                true_to_pred[tlabel][idx] = cnt

        for tlabel in sorted(true_counts.keys()):
            counts = true_to_pred.get(tlabel, {})
            self.stdout.write(f'  {tlabel} (n={true_counts[tlabel]}): {dict(counts)}')

        # Suggest creating a classes_order.json if mapping covers all indices 0..N-1
        if suggested:
            max_idx = max(suggested.keys())
            # propose an ordering of length max_idx+1 using suggested mapping where available
            proposed = [None] * (max_idx + 1)
            for i in range(len(proposed)):
                if i in suggested:
                    proposed[i] = suggested[i]
                else:
                    proposed[i] = f'index_{i}'

            self.stdout.write('\nSuggested classes_order (length=%d):' % len(proposed))
            self.stdout.write(str(proposed))
            self.stdout.write('\nIf this mapping looks correct, you can save it as media/models/classes_order.json and restart the server to apply the correct label mapping.')

        self.stdout.write(self.style.SUCCESS('Diagnosis finished.'))
