from math import isfinite
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

AXES = ['time', 'transfer', 'crowd', 'walk', 'weakAc', 'familiarRoute', 'fare', 'elevator', 'stairs', 'express']
CODES = ['FAST', 'FEW_TRANSFERS', 'UNCROWDED', 'LESS_WALK', 'WEAK_AC', 'FAMILIAR_ROUTE', 'LOW_FARE', 'ELEVATOR', 'AVOID_STAIRS', 'EXPRESS']
VERSION = 'route-preference-v2'

class ChoiceSnapshot(BaseModel):
    model_config = ConfigDict(extra='forbid')
    selected: Dict[str, float]
    alternatives: List[Dict[str, float]] = Field(default_factory=list, max_length=500)

    @field_validator('selected', 'alternatives')
    @classmethod
    def finite_features(cls, value):
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if any(key not in AXES or not isfinite(number) or number < 0 for key, number in row.items()):
                raise ValueError('Unknown axis or invalid feature')
        return value

class PreferenceRequestV2(BaseModel):
    model_config = ConfigDict(extra='forbid')
    routePrefs: List[str] = Field(default_factory=list, max_length=32)
    choices: List[ChoiceSnapshot] = Field(default_factory=list, max_length=100)

class PreferenceResponseV2(BaseModel):
    modelVersion: Literal['route-preference-v2'] = VERSION
    axisKeys: List[str] = Field(default_factory=lambda: list(AXES))
    vector: List[float]
    observationCounts: List[int]
    confidence: List[float]
    source: Literal['AI'] = 'AI'

def compute_preference_v2(request: PreferenceRequestV2) -> PreferenceResponseV2:
    """Compare chosen and offered routes. Missing/unchanging axes carry no behavioral evidence."""
    vector, counts, confidences = [], [], []
    higher_is_better = {'weakAc', 'familiarRoute', 'elevator', 'express'}
    for axis, code in zip(AXES, CODES):
        observations = []
        for choice in request.choices:
            selected = choice.selected.get(axis)
            if selected is None:
                continue
            values = [selected] + [alternative[axis] for alternative in choice.alternatives if axis in alternative]
            low, high = min(values), max(values)
            if high <= low:
                continue
            relative = (selected - low) / (high - low)
            observations.append(relative if axis in higher_is_better else 1 - relative)
        count = len(observations)
        # Versioned product default: five comparable choices give 50% behavioral weight.
        confidence = count / (count + 5)
        survey = 1.0 if code in request.routePrefs else 0.5
        observed = sum(observations) / count if count else survey
        vector.append(round(survey * (1 - confidence) + observed * confidence, 6))
        counts.append(count)
        confidences.append(round(confidence, 6))
    return PreferenceResponseV2(vector=vector, observationCounts=counts, confidence=confidences)
