"""The 5 quality dimensions, shared across the pipeline.

Kept dependency-free (no torch) so scripts that only need the dimension
names -- e.g. generate_weight_combinations.py -- don't have to import
torch just to read this list.
"""

DIMENSIONS = [
    "educational_value",
    "reasoning",
    "professionalism",
    "cleanliness",
    "cultural_nuance",
]
