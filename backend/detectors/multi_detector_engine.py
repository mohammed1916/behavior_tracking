"""Multi-detector engine for combining outputs from multiple detectors"""

import logging
from typing import Dict, List, Optional
from enum import Enum

from .base import DetectorOutput, DetectorBase

logger = logging.getLogger(__name__)


class MultiDetectorMode(Enum):
    """How to combine detector outputs"""
    CONSENSUS = "consensus"      # All must agree
    WEIGHTED = "weighted"        # Weighted average by detector confidence
    CASCADE = "cascade"          # Use primary, fall back to secondary
    VETO = "veto"               # Primary with override capability
    MAJORITY = "majority"        # >50% of detectors must agree


class MultiDetectorEngine:
    """Intelligently fuse outputs from multiple detectors"""
    
    # Learned weights for each detector (tuned via validation)
    DEFAULT_WEIGHTS = {
        'mediapipe': 0.4,      # Motion is a strong signal
        'yolo': 0.3,           # Object presence is supporting signal
        'vlm': 0.3,            # Semantic reasoning is flexible
    }
    
    def __init__(
        self,
        detectors: Dict[str, DetectorBase],
        mode: MultiDetectorMode = MultiDetectorMode.WEIGHTED,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-detector engine.
        
        Args:
            detectors: Dict of {detector_name: DetectorBase}
            mode: How to combine detector outputs
            weights: Optional custom weights for weighted mode
        """
        self.detectors = detectors
        self.mode = mode if isinstance(mode, MultiDetectorMode) else MultiDetectorMode[mode.upper()]
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights so they sum to 1.0"""
        active_detectors = {k: v for k, v in self.weights.items() if k in self.detectors}
        if not active_detectors:
            return
        
        total = sum(active_detectors.values())
        if total > 0:
            for key in active_detectors:
                self.weights[key] = self.weights[key] / total
    
    def fuse(self, outputs: Dict[str, DetectorOutput]) -> DetectorOutput:
        """
        Fuse detector outputs into a single decision.
        
        Args:
            outputs: Dict of {detector_name: DetectorOutput}
        
        Returns:
            Fused DetectorOutput
        """
        if not outputs:
            return DetectorOutput('unknown', 0.0, {})
        
        if self.mode == MultiDetectorMode.CONSENSUS:
            return self._consensus(outputs)
        elif self.mode == MultiDetectorMode.WEIGHTED:
            return self._weighted(outputs)
        elif self.mode == MultiDetectorMode.CASCADE:
            return self._cascade(outputs)
        elif self.mode == MultiDetectorMode.VETO:
            return self._veto(outputs)
        elif self.mode == MultiDetectorMode.MAJORITY:
            return self._majority(outputs)
        else:
            return self._weighted(outputs)  # Default fallback
    
    def _consensus(self, outputs: Dict[str, DetectorOutput]) -> DetectorOutput:
        """All detectors must agree on label"""
        if not outputs:
            return DetectorOutput('unknown', 0.0, {})
        
        labels = [o.label for o in outputs.values() if o.label != 'unknown']
        
        if not labels:
            # No confident predictions
            return DetectorOutput('unknown', 0.0, {'reason': 'no_agreement'})
        
        # Check if all are the same
        if len(set(labels)) == 1:
            agreed_label = labels[0]
            avg_conf = sum(o.confidence for o in outputs.values()) / len(outputs)
            return DetectorOutput(
                agreed_label,
                avg_conf,
                {'mode': 'consensus', 'consensus': True, 'detectors': list(outputs.keys())}
            )
        else:
            # Disagreement - return unknown or most confident
            most_confident = max(outputs.values(), key=lambda x: x.confidence)
            return DetectorOutput(
                'unknown',
                0.0,
                {'reason': 'no_consensus', 'detectors': list(outputs.keys())}
            )
    
    def _weighted(self, outputs: Dict[str, DetectorOutput]) -> DetectorOutput:
        """Weighted average of detector confidence scores"""
        if not outputs:
            return DetectorOutput('unknown', 0.0, {})
        
        # Separate work/idle predictions
        work_score = 0.0
        idle_score = 0.0
        total_weight = 0.0
        
        for detector_name, output in outputs.items():
            weight = self.weights.get(detector_name, 1.0 / len(outputs))
            
            if output.label == 'work':
                work_score += output.confidence * weight
            elif output.label == 'idle':
                idle_score += output.confidence * weight
            # unknown votes are ignored
        
        # Normalize
        total_score = work_score + idle_score
        if total_score == 0:
            return DetectorOutput('unknown', 0.0, {'reason': 'no_predictions'})
        
        work_score /= total_score
        idle_score /= total_score
        
        # Decide based on which score is higher
        if work_score > idle_score:
            label = 'work'
            confidence = work_score
        else:
            label = 'idle'
            confidence = idle_score
        
        return DetectorOutput(
            label,
            confidence,
            {
                'mode': 'weighted',
                'work_score': work_score,
                'idle_score': idle_score,
                'detectors': list(outputs.keys())
            }
        )
    
    def _cascade(self, outputs: Dict[str, DetectorOutput]) -> DetectorOutput:
        """
        Use primary detector (e.g., MediaPipe), fall back to secondary if uncertain.
        Useful when VLM is expensive and we want to use it only when other detectors disagree.
        """
        # Define priority order
        priority_order = ['mediapipe', 'yolo', 'vlm']
        
        for detector_name in priority_order:
            if detector_name in outputs:
                output = outputs[detector_name]
                if output.confidence > 0.6 or output.label != 'unknown':
                    return DetectorOutput(
                        output.label,
                        output.confidence,
                        {'mode': 'cascade', 'primary_detector': detector_name}
                    )
        
        # Fallback: use most confident detector
        if outputs:
            most_confident = max(outputs.values(), key=lambda x: x.confidence)
            return most_confident
        
        return DetectorOutput('unknown', 0.0, {'reason': 'cascade_no_match'})
    
    def _veto(self, outputs: Dict[str, DetectorOutput]) -> DetectorOutput:
        """
        Use MediaPipe + YOLO fusion, but allow VLM to veto low-confidence decisions.
        Useful for final verification.
        """
        # Fuse MediaPipe + YOLO
        low_level = {}
        if 'mediapipe' in outputs:
            low_level['mediapipe'] = outputs['mediapipe']
        if 'yolo' in outputs:
            low_level['yolo'] = outputs['yolo']
        
        if low_level:
            low_level_output = self._weighted(low_level)
        else:
            low_level_output = None
        
        # Check VLM veto
        vlm_output = outputs.get('vlm')
        
        if vlm_output and vlm_output.confidence > 0.7:
            # VLM has high confidence - override
            return DetectorOutput(
                vlm_output.label,
                vlm_output.confidence,
                {'mode': 'veto', 'vetoed_by': 'vlm'}
            )
        elif low_level_output:
            # No veto, use low-level fusion
            return DetectorOutput(
                low_level_output.label,
                low_level_output.confidence,
                {'mode': 'veto', 'vetoed_by': None}
            )
        else:
            return DetectorOutput('unknown', 0.0, {'reason': 'veto_no_detectors'})
    
    def _majority(self, outputs: Dict[str, DetectorOutput]) -> DetectorOutput:
        """Vote-based: >50% of detectors must agree"""
        labels = [o.label for o in outputs.values() if o.label != 'unknown']
        
        if not labels:
            return DetectorOutput('unknown', 0.0, {'reason': 'no_votes'})
        
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find label with most votes
        winning_label = max(label_counts.items(), key=lambda x: x[1])[0]
        winning_votes = label_counts[winning_label]
        total_votes = len(labels)
        
        if winning_votes > total_votes * 0.5:
            # Majority reached
            avg_conf = sum(o.confidence for o in outputs.values() if o.label == winning_label) / winning_votes
            return DetectorOutput(
                winning_label,
                avg_conf,
                {'mode': 'majority', 'votes': label_counts}
            )
        else:
            return DetectorOutput('unknown', 0.0, {'reason': 'no_majority', 'votes': label_counts})
