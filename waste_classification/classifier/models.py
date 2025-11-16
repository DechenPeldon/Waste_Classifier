from django.db import models
from django.utils import timezone

class ClassificationResult(models.Model):
    image = models.ImageField(upload_to='uploads/')
    # Allow null/blank so we can save a missing prediction instead of a placeholder string
    predicted_class = models.CharField(max_length=100, null=True, blank=True)
    confidence = models.FloatField()
    is_camera = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        # Ensure a readable string even if predicted_class is empty
        label = self.predicted_class if self.predicted_class else 'Unknown'
        return f"{label} - {self.confidence:.2f}%"
    
    def get_waste_description(self):
        # Attempt to map the predicted label to a maintained descriptions mapping.
        # Prefer descriptions for labels declared in `ml_model/labels.txt` when present.
        from pathlib import Path

        # Predefined description templates for common classes (keys are normalized lowercased labels)
        templates = {
            'e-waste': {
                'description': 'Electronic waste includes discarded electrical or electronic devices. These items contain valuable materials that can be recovered and hazardous substances that need proper disposal.',
                'examples': 'Old computers, mobile phones, televisions, printers, keyboards, circuit boards',
                'disposal': 'Take to certified e-waste recycling centers. Never throw in regular trash. Many retailers offer take-back programs.',
                'recyclable': True
            },
            'automobile': {
                'description': 'Automotive waste includes parts and materials from vehicles. Many components contain metals and fluids that require special handling.',
                'examples': 'Car parts, tires, batteries, oil filters, brake pads, bumpers',
                'disposal': 'Contact auto recyclers or scrap yards. Some parts can be refurbished. Fluids must be disposed at hazardous waste facilities.',
                'recyclable': True
            },
            'battery': {
                'description': 'Batteries contain toxic chemicals and heavy metals that can harm the environment if improperly disposed. They require specialized recycling.',
                'examples': 'AA/AAA batteries, lithium-ion batteries, car batteries, rechargeable batteries',
                'disposal': 'Return to battery retailers or designated collection points. Never incinerate or puncture batteries.',
                'recyclable': True
            },
            'glass': {
                'description': 'Glass is 100% recyclable and can be recycled endlessly without loss in quality. It saves energy and reduces landfill waste.',
                'examples': 'Bottles, jars, containers, broken glass, window glass',
                'disposal': 'Rinse and place in glass recycling bin. Remove lids and caps. Some areas accept mixed colors, others require separation.',
                'recyclable': True
            },
            'light-bulbs': {
                'description': 'Light bulbs, especially CFLs and fluorescents, contain mercury and require special handling. LED bulbs are safer but still need proper recycling.',
                'examples': 'CFL bulbs, LED bulbs, fluorescent tubes, incandescent bulbs, halogen bulbs',
                'disposal': 'Take to hardware stores or hazardous waste facilities. Do not break. Some retailers offer free recycling programs.',
                'recyclable': True
            },
            'metal': {
                'description': 'Metals are highly valuable recyclables. Recycling metal saves significant energy compared to producing new metal from ore.',
                'examples': 'Aluminum cans, steel cans, copper wire, brass fixtures, iron scrap, tin containers',
                'disposal': 'Clean and place in metal recycling bin. Scrap yards accept larger items. Separate ferrous and non-ferrous if required.',
                'recyclable': True
            },
            'organic': {
                'description': 'Organic waste can be composted to create nutrient-rich soil. Composting reduces methane emissions from landfills.',
                'examples': 'Food scraps, fruit peels, vegetable waste, coffee grounds, tea bags, yard trimmings, leaves',
                'disposal': 'Compost at home or use municipal composting services. Avoid meat and dairy in home compost. Great for gardens.',
                'recyclable': True
            },
            'paper': {
                'description': 'Paper is one of the most recycled materials. Recycling paper saves trees, energy, and water while reducing greenhouse gases.',
                'examples': 'Newspapers, magazines, cardboard, office paper, books, paper bags, cereal boxes',
                'disposal': 'Keep dry and clean. Remove plastic windows from envelopes. Flatten cardboard boxes. Place in paper recycling bin.',
                'recyclable': True
            },
            'plastic': {
                'description': 'Plastic recycling helps reduce petroleum consumption and landfill waste. Check local recycling codes as not all plastics are accepted everywhere.',
                'examples': 'Bottles, containers, bags, packaging, plastic utensils, toys, straws',
                'disposal': 'Check recycling number (1-7). Rinse containers. Remove caps if required. Reduce single-use plastics when possible.',
                'recyclable': True
            }
        }

        # Resolve labels file (if present) and try to match the predicted label
        labels_path = Path(__file__).resolve().parent / 'ml_model' / 'labels.txt'
        available_labels = []
        try:
            if labels_path.exists():
                available_labels = [l.strip() for l in labels_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        except Exception:
            available_labels = []

        label = (self.predicted_class or '').strip()
        if not label:
            return {}

        # Normalize label for lookup: remove words like 'waste'/'wastes', strip numeric prefixes
        import re

        # remove leading digits and separators (if any)
        label_no_index = re.sub(r'^\s*\d+\s*[\.\-:)]*\s*', '', label)
        # remove 'waste' words to match template keys (e.g. 'paper waste' -> 'paper')
        label_no_waste = re.sub(r'\b(waste|wastes)\b', '', label_no_index, flags=re.IGNORECASE).strip()
        lookup_key = label_no_waste.lower()
        # Direct match in templates
        if lookup_key in templates:
            return templates[lookup_key]

        # Try normalized variants (spaces vs hyphens)
        alt = lookup_key.replace(' ', '-').replace('_', '-')
        if alt in templates:
            return templates[alt]

        # If the label exists in labels.txt but we don't have a specific template, provide a safe generic description
        # Normalize available_labels similarly for detection
        norm_available = []
        for l in available_labels:
            l_clean = re.sub(r'^\s*\d+\s*[\.\-:)]*\s*', '', l).strip()
            l_clean = re.sub(r'\b(waste|wastes)\b', '', l_clean, flags=re.IGNORECASE).strip()
            norm_available.append(l_clean.lower())

        if lookup_key in norm_available or alt in norm_available or label_no_index.lower() in norm_available:
            return {
                'description': f'No app-specific description available for "{label}". The model recognizes this class — consider adding a tailored description to the app.',
                'examples': '',
                'disposal': 'Refer to local waste management guidance for disposal instructions for this item.',
                'recyclable': False
            }

        # Nothing matched
        return {}