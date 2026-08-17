"""
Conformational Clustering Module

Converts MD trajectories or structural ensembles into discrete states
with population estimates for CDST training.

Methods:
1. RMSD-based K-means clustering
2. Hierarchical clustering with automatic K selection
3. tICA + clustering for kinetics-aware state decomposition
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings


@dataclass
class ClusteringResult:
    """Result of conformational clustering."""
    n_states: int
    assignments: np.ndarray      # (N_frames,) state assignment per frame
    populations: np.ndarray      # (K,) population of each state
    centroids: np.ndarray        # (K, n_features) cluster centroids
    method: str
    silhouette: float            # Clustering quality metric
    inertia: Optional[float] = None  # K-means inertia


class ConformationalClusterer:
    """Cluster conformations into discrete states.
    
    Supports multiple clustering strategies and automatic K selection.
    """
    
    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        max_clusters: int = 10,
        random_state: int = 42,
    ):
        """
        Args:
            method: 'kmeans', 'hierarchical', or 'auto'
            n_clusters: Fixed number of clusters (if None, auto-select)
            max_clusters: Maximum K to try in auto-selection
            random_state: Random seed
        """
        self.method = method
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
    
    def fit(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> ClusteringResult:
        """Cluster conformations.
        
        Args:
            features: (N_frames, n_features) conformational features
                     (e.g., flattened coordinates, distances, dihedrals)
            weights: (N_frames,) optional frame weights
        
        Returns:
            ClusteringResult with assignments and populations
        """
        if self.n_clusters is not None:
            return self._fit_fixed_k(features, self.n_clusters, weights)
        else:
            return self._fit_auto_k(features, weights)
    
    def _fit_fixed_k(
        self, 
        features: np.ndarray, 
        k: int,
        weights: Optional[np.ndarray] = None,
    ) -> ClusteringResult:
        """Cluster with fixed K."""
        if self.method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            assignments = model.fit_predict(features)
            centroids = model.cluster_centers_
            inertia = model.inertia_
        elif self.method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=k)
            assignments = model.fit_predict(features)
            # Compute centroids manually
            centroids = np.array([features[assignments == i].mean(axis=0) for i in range(k)])
            inertia = None
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute populations; honour frame weights when supplied so that
        # non-uniform subsampling cannot silently bias state fractions.
        if weights is not None:
            weighted = np.bincount(assignments, weights=weights, minlength=k)
            populations = weighted / weights.sum()
        else:
            populations = np.bincount(assignments, minlength=k) / len(assignments)

        # Compute silhouette score (if k > 1). The subsample draw must be
        # seeded, otherwise auto-K selection compares irreproducible scores.
        if k > 1 and len(features) > k:
            sil = silhouette_score(features, assignments,
                                   sample_size=min(1000, len(features)),
                                   random_state=self.random_state)
        else:
            sil = 0.0
        
        return ClusteringResult(
            n_states=k,
            assignments=assignments,
            populations=populations,
            centroids=centroids,
            method=self.method,
            silhouette=sil,
            inertia=inertia,
        )
    
    def _fit_auto_k(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> ClusteringResult:
        """Automatically select K using silhouette score."""
        best_result = None
        best_silhouette = -1
        
        for k in range(2, self.max_clusters + 1):
            result = self._fit_fixed_k(features, k, weights)
            if result.silhouette > best_silhouette:
                best_silhouette = result.silhouette
                best_result = result
        
        return best_result
    
    @staticmethod
    def compute_rmsd_features(coordinates: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute RMSD-based features from coordinates.

        Each frame is superposed onto the reference by Kabsch alignment
        before flattening, so that downstream clustering sees conformational
        difference rather than global rotation/translation.

        Args:
            coordinates: (N_frames, n_atoms, 3) atomic coordinates
            reference: (n_atoms, 3) reference structure for alignment

        Returns:
            features: (N_frames, n_atoms*3) flattened aligned coordinates
        """
        n_frames, n_atoms, _ = coordinates.shape

        if reference is None:
            reference = coordinates.mean(axis=0)

        ref_centered = reference - reference.mean(axis=0)

        aligned = np.empty_like(coordinates)
        for i in range(n_frames):
            frame = coordinates[i] - coordinates[i].mean(axis=0)
            # Standard Kabsch: SVD of the cross-covariance with a
            # reflection correction gives the optimal rotation mapping the
            # frame onto the reference.
            u, _, vt = np.linalg.svd(frame.T @ ref_centered)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
            aligned[i] = frame @ rotation.T

        return aligned.reshape(n_frames, -1)
    
    @staticmethod
    def compute_contact_features(coordinates: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
        """Compute contact map features.
        
        Args:
            coordinates: (N_frames, n_atoms, 3) atomic coordinates
            cutoff: Distance cutoff for contacts (Angstrom)
        
        Returns:
            features: (N_frames, n_contacts) binary contact features
        """
        n_frames, n_atoms, _ = coordinates.shape
        
        # Compute pairwise distances for first frame to identify contact pairs
        ref = coordinates[0]
        dist_matrix = np.sqrt(((ref[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1))
        contact_pairs = np.argwhere(np.triu(dist_matrix < cutoff, k=1))
        
        # Compute contacts for all frames
        features = np.zeros((n_frames, len(contact_pairs)))
        for i, (a, b) in enumerate(contact_pairs):
            dists = np.sqrt(((coordinates[:, a, :] - coordinates[:, b, :]) ** 2).sum(axis=-1))
            features[:, i] = (dists < cutoff).astype(float)
        
        return features


def cluster_ensemble(
    ensemble_coords: np.ndarray,
    n_states: Optional[int] = None,
    method: str = 'kmeans',
    feature_type: str = 'coordinates',
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to cluster an ensemble.
    
    Args:
        ensemble_coords: (N_frames, n_atoms, 3) coordinates
        n_states: Number of states (None for auto)
        method: Clustering method
        feature_type: 'coordinates' or 'contacts'
    
    Returns:
        assignments: (N_frames,) state assignments
        populations: (K,) state populations
    """
    # Compute features
    if feature_type == 'coordinates':
        features = ConformationalClusterer.compute_rmsd_features(ensemble_coords)
    elif feature_type == 'contacts':
        features = ConformationalClusterer.compute_contact_features(ensemble_coords)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # Cluster
    clusterer = ConformationalClusterer(method=method, n_clusters=n_states)
    result = clusterer.fit(features)
    
    return result.assignments, result.populations


def assign_to_reference_states(
    ensemble_coords: np.ndarray,
    reference_coords: np.ndarray,
    rmsd_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign conformations to predefined reference states.
    
    Used when reference structures for each state are known
    (e.g., from crystal structures or previous clustering).
    
    Args:
        ensemble_coords: (N_frames, n_atoms, 3) ensemble coordinates
        reference_coords: (K, n_atoms, 3) reference structure per state
        rmsd_threshold: Maximum RMSD for assignment (Angstrom)
    
    Returns:
        assignments: (N_frames,) state assignments (-1 for unassigned)
        populations: (K,) state populations
    """
    n_frames = ensemble_coords.shape[0]
    n_states = reference_coords.shape[0]
    
    # Compute RMSD to each reference
    assignments = np.full(n_frames, -1, dtype=int)
    min_rmsds = np.full(n_frames, np.inf)
    
    for k in range(n_states):
        ref = reference_coords[k]
        # Compute RMSD (simplified - assumes pre-aligned)
        rmsds = np.sqrt(((ensemble_coords - ref) ** 2).sum(axis=(1, 2)) / ensemble_coords.shape[1])
        
        # Assign if closer than current assignment and within threshold
        closer = rmsds < min_rmsds
        within_threshold = rmsds < rmsd_threshold
        update = closer & within_threshold
        
        assignments[update] = k
        min_rmsds[update] = rmsds[update]
    
    # Compute populations (excluding unassigned)
    assigned = assignments >= 0
    populations = np.zeros(n_states)
    if assigned.sum() > 0:
        for k in range(n_states):
            populations[k] = (assignments == k).sum() / assigned.sum()
    
    return assignments, populations


def compute_state_coverage(
    apo_coords: np.ndarray,
    holo_coords: np.ndarray,
    rmsd_threshold: float = 3.0,
) -> float:
    """Compute State Coverage (SC) metric.
    
    SC = fraction of holo conformations covered by apo states.
    
    Args:
        apo_coords: (N_apo, n_atoms, 3) apo ensemble
        holo_coords: (N_holo, n_atoms, 3) holo ensemble
        rmsd_threshold: RMSD threshold for coverage
    
    Returns:
        SC: State Coverage value in [0, 1]
    """
    n_holo = holo_coords.shape[0]
    covered = 0
    
    for i in range(n_holo):
        holo_conf = holo_coords[i]
        # Compute RMSD to all apo conformations
        rmsds = np.sqrt(((apo_coords - holo_conf) ** 2).sum(axis=(1, 2)) / apo_coords.shape[1])
        if rmsds.min() < rmsd_threshold:
            covered += 1
    
    return covered / n_holo
