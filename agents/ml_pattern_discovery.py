"""
Machine Learning Pattern Discovery Module
Uses unsupervised learning to discover unnamed patterns in market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from agents.base_agent import BaseAgent
from utils.logger import setup_logger
from utils.slack_notifier import slack_notifier

class MLPatternDiscovery(BaseAgent):
    """
    Discovers trading patterns using machine learning techniques
    """
    
    def __init__(self):
        """Initialize ML pattern discovery"""
        super().__init__('MLPatternDiscovery')
        self.logger = setup_logger('MLPatternDiscovery')
        
        # Feature engineering parameters
        self.feature_windows = [5, 10, 20, 50]  # Different lookback periods
        self.n_clusters = 20  # Default number of clusters
        
        # Discovered patterns storage
        self.ml_patterns = []
        self.pattern_clusters = None
        self.scaler = StandardScaler()
        
        self.logger.info("🤖 ML Pattern Discovery initialized")
    
    async def initialize(self) -> bool:
        """Initialize the ML discovery system"""
        try:
            self.logger.info("Machine learning pattern discovery ready")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, data: pd.DataFrame) -> List[Dict]:
        """
        Discover patterns using ML
        
        Args:
            data: Market data
        
        Returns:
            List[Dict]: Discovered ML patterns
        """
        return await self.discover_ml_patterns(data)
    
    async def discover_ml_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Main ML pattern discovery pipeline
        
        Args:
            data: Historical market data
        
        Returns:
            List[Dict]: Discovered patterns
        """
        self.logger.info("🔬 Starting ML pattern discovery...")
        
        try:
            # 1. Feature engineering
            features, feature_names = self.engineer_features(data)
            
            if features is None or len(features) == 0:
                self.logger.warning("No features extracted")
                return []
            
            # 2. Dimensionality reduction
            reduced_features = self.reduce_dimensions(features)
            
            # 3. Clustering
            clusters = self.cluster_patterns(reduced_features)
            
            # 4. Analyze clusters
            patterns = self.analyze_clusters(clusters, features, data, feature_names)
            
            # 5. Find anomalies (rare but profitable setups)
            anomaly_patterns = self.find_anomaly_patterns(features, data)
            patterns.extend(anomaly_patterns)
            
            # 6. Sequence mining (find repeating sequences)
            sequence_patterns = self.mine_sequences(data)
            patterns.extend(sequence_patterns)
            
            self.logger.info(f"✨ Discovered {len(patterns)} ML patterns")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in ML pattern discovery: {e}")
            return []
    
    def engineer_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer features for pattern discovery
        
        Args:
            data: Market data
        
        Returns:
            Tuple[np.ndarray, List[str]]: Feature matrix and feature names
        """
        self.logger.info("🔧 Engineering features...")
        
        features_list = []
        feature_names = []
        
        try:
            # Price-based features
            for window in self.feature_windows:
                if len(data) < window:
                    continue
                
                # Returns
                returns = data['Close'].pct_change(window)
                features_list.append(returns.fillna(0))
                feature_names.append(f'return_{window}')
                
                # Volatility
                volatility = data['Close'].rolling(window).std() / data['Close'].rolling(window).mean()
                features_list.append(volatility.fillna(0))
                feature_names.append(f'volatility_{window}')
                
                # Price position in range
                high_roll = data['High'].rolling(window).max()
                low_roll = data['Low'].rolling(window).min()
                price_position = (data['Close'] - low_roll) / (high_roll - low_roll + 0.0001)
                features_list.append(price_position.fillna(0.5))
                feature_names.append(f'price_position_{window}')
                
                # Trend strength
                x = np.arange(window)
                trend_strength = pd.Series(index=data.index, dtype=float)
                for i in range(window, len(data)):
                    y = data['Close'].iloc[i-window:i].values
                    if len(y) == window:
                        slope = np.polyfit(x, y, 1)[0]
                        trend_strength.iloc[i] = slope / np.mean(y)
                features_list.append(trend_strength.fillna(0))
                feature_names.append(f'trend_strength_{window}')
            
            # Volume features (if available)
            if 'Volume' in data.columns:
                # Volume ratio
                volume_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
                features_list.append(volume_ratio.fillna(1))
                feature_names.append('volume_ratio')
                
                # Volume trend
                volume_change = data['Volume'].pct_change(5)
                features_list.append(volume_change.fillna(0))
                feature_names.append('volume_change')
            
            # Technical indicators (if available)
            if 'RSI' in data.columns:
                # RSI levels
                rsi_norm = (data['RSI'] - 50) / 50
                features_list.append(rsi_norm.fillna(0))
                feature_names.append('rsi_normalized')
                
                # RSI divergence
                price_change = data['Close'].pct_change(10)
                rsi_change = data['RSI'].pct_change(10)
                divergence = price_change - rsi_change
                features_list.append(divergence.fillna(0))
                feature_names.append('rsi_divergence')
            
            # Candlestick patterns
            candle_features = self.extract_candlestick_features(data)
            for name, feature in candle_features.items():
                features_list.append(feature)
                feature_names.append(name)
            
            # Market microstructure
            spread = (data['High'] - data['Low']) / data['Close']
            features_list.append(spread.fillna(0))
            feature_names.append('spread')
            
            gap = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            features_list.append(gap.fillna(0))
            feature_names.append('gap')
            
            # Combine all features
            if features_list:
                features_df = pd.DataFrame({name: feat for name, feat in zip(feature_names, features_list)})
                features_df = features_df.dropna()
                
                self.logger.info(f"  Extracted {len(feature_names)} features")
                
                return features_df.values, feature_names
            else:
                return None, []
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            return None, []
    
    def extract_candlestick_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Extract candlestick pattern features
        
        Args:
            data: Market data
        
        Returns:
            Dict[str, pd.Series]: Candlestick features
        """
        features = {}
        
        # Body size
        body_size = abs(data['Close'] - data['Open']) / data['Close']
        features['body_size'] = body_size.fillna(0)
        
        # Upper shadow
        upper_shadow = (data['High'] - data[['Open', 'Close']].max(axis=1)) / data['Close']
        features['upper_shadow'] = upper_shadow.fillna(0)
        
        # Lower shadow  
        lower_shadow = (data[['Open', 'Close']].min(axis=1) - data['Low']) / data['Close']
        features['lower_shadow'] = lower_shadow.fillna(0)
        
        # Doji detection
        doji = (body_size < 0.001).astype(float)
        features['is_doji'] = doji
        
        # Hammer detection
        hammer = ((lower_shadow > body_size * 2) & (upper_shadow < body_size * 0.5)).astype(float)
        features['is_hammer'] = hammer
        
        # Engulfing pattern
        prev_body = abs(data['Close'].shift(1) - data['Open'].shift(1))
        engulfing = (body_size > prev_body * 1.5).astype(float)
        features['is_engulfing'] = engulfing
        
        return features
    
    def reduce_dimensions(self, features: np.ndarray, n_components: int = 10) -> np.ndarray:
        """
        Reduce feature dimensions using PCA
        
        Args:
            features: Feature matrix
            n_components: Number of components
        
        Returns:
            np.ndarray: Reduced features
        """
        self.logger.info(f"📉 Reducing dimensions from {features.shape[1]} to {n_components}")
        
        try:
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Apply PCA
            n_components = min(n_components, features.shape[0], features.shape[1])
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(features_scaled)
            
            explained_variance = sum(pca.explained_variance_ratio_)
            self.logger.info(f"  PCA explained variance: {explained_variance:.1%}")
            
            return reduced
            
        except Exception as e:
            self.logger.error(f"Error in dimensionality reduction: {e}")
            return features
    
    def cluster_patterns(self, features: np.ndarray) -> np.ndarray:
        """
        Cluster similar patterns
        
        Args:
            features: Feature matrix
        
        Returns:
            np.ndarray: Cluster labels
        """
        self.logger.info(f"🎯 Clustering patterns into {self.n_clusters} groups")
        
        try:
            # KMeans clustering
            kmeans = KMeans(n_clusters=min(self.n_clusters, len(features)), 
                           random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # Store for later use
            self.pattern_clusters = kmeans
            
            # Log cluster distribution
            unique, counts = np.unique(clusters, return_counts=True)
            self.logger.info(f"  Cluster sizes: {dict(zip(unique, counts))}")
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in clustering: {e}")
            return np.zeros(len(features))
    
    def analyze_clusters(self, clusters: np.ndarray, features: np.ndarray, 
                        data: pd.DataFrame, feature_names: List[str]) -> List[Dict]:
        """
        Analyze clusters to create pattern definitions
        
        Args:
            clusters: Cluster labels
            features: Feature matrix
            data: Original market data
            feature_names: Names of features
        
        Returns:
            List[Dict]: Pattern definitions
        """
        self.logger.info("🔍 Analyzing clusters for profitable patterns")
        
        patterns = []
        
        try:
            unique_clusters = np.unique(clusters)
            
            for cluster_id in unique_clusters:
                # Get cluster members
                cluster_mask = clusters == cluster_id
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) < 10:  # Skip small clusters
                    continue
                
                # Analyze cluster profitability
                profitability = self.analyze_cluster_profitability(cluster_mask, data)
                
                if profitability['win_rate'] < 0.5:  # Skip unprofitable clusters
                    continue
                
                # Create pattern definition
                pattern = {
                    'name': f'ML_Pattern_Cluster_{cluster_id}',
                    'type': 'ml_discovered',
                    'cluster_id': int(cluster_id),
                    'discovery_method': 'kmeans_clustering',
                    
                    # Feature characteristics
                    'feature_means': {
                        name: float(np.mean(cluster_features[:, i]))
                        for i, name in enumerate(feature_names[:cluster_features.shape[1]])
                    },
                    'feature_stds': {
                        name: float(np.std(cluster_features[:, i]))
                        for i, name in enumerate(feature_names[:cluster_features.shape[1]])
                    },
                    
                    # Entry conditions (based on feature similarity)
                    'entry_conditions': {
                        'cluster_distance_threshold': 0.2,  # Distance from cluster center
                        'min_feature_match': 0.7,  # Minimum feature similarity
                        'feature_weights': self.calculate_feature_importance(cluster_features, feature_names)
                    },
                    
                    # Exit conditions (statistical)
                    'exit_conditions': {
                        'holding_period': profitability['optimal_holding'],
                        'profit_target': profitability['avg_profit'] * 1.5,
                        'stop_loss': profitability['avg_loss'] * 1.5
                    },
                    
                    # Filters
                    'filters': {
                        'min_cluster_confidence': 0.7,
                        'market_regime': profitability['best_regime']
                    },
                    
                    # Statistics
                    'statistics': {
                        'occurrences': int(np.sum(cluster_mask)),
                        'preliminary_win_rate': profitability['win_rate'],
                        'avg_return': profitability['avg_return'],
                        'best_time': profitability['best_time']
                    },
                    
                    'confidence': profitability['confidence']
                }
                
                patterns.append(pattern)
                self.logger.info(f"  Found ML pattern: Cluster {cluster_id} (WR: {profitability['win_rate']:.1%})")
                
                # Send Slack notification for ML pattern
                import asyncio
                asyncio.create_task(slack_notifier.ml_pattern_found(
                    'clustering',
                    int(cluster_id),
                    {
                        'occurrences': int(np.sum(cluster_mask)),
                        'win_rate': profitability['win_rate'],
                        'confidence': profitability['confidence']
                    }
                ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing clusters: {e}")
            return []
    
    def analyze_cluster_profitability(self, cluster_mask: np.ndarray, 
                                     data: pd.DataFrame) -> Dict:
        """
        Analyze profitability of a cluster
        
        Args:
            cluster_mask: Boolean mask for cluster members
            data: Market data
        
        Returns:
            Dict: Profitability metrics
        """
        # Get indices where pattern occurs
        pattern_indices = np.where(cluster_mask)[0]
        
        results = []
        for idx in pattern_indices:
            if idx < len(data) - 10:  # Need future data
                # Entry at pattern occurrence
                entry_price = data['Close'].iloc[idx]
                
                # Check various exit points
                for exit_offset in [1, 2, 3, 5, 10]:
                    if idx + exit_offset < len(data):
                        exit_price = data['Close'].iloc[idx + exit_offset]
                        return_pct = (exit_price - entry_price) / entry_price
                        
                        results.append({
                            'holding_period': exit_offset,
                            'return': return_pct,
                            'win': return_pct > 0,
                            'entry_idx': idx
                        })
        
        if not results:
            return {
                'win_rate': 0,
                'avg_return': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'optimal_holding': 5,
                'best_regime': 'unknown',
                'best_time': 'any',
                'confidence': 0
            }
        
        # Calculate metrics
        results_df = pd.DataFrame(results)
        
        # Best holding period
        holding_performance = results_df.groupby('holding_period').agg({
            'win': 'mean',
            'return': 'mean'
        })
        optimal_holding = holding_performance['return'].idxmax()
        
        # Overall metrics
        best_results = results_df[results_df['holding_period'] == optimal_holding]
        
        wins = best_results[best_results['win']]
        losses = best_results[~best_results['win']]
        
        return {
            'win_rate': best_results['win'].mean(),
            'avg_return': best_results['return'].mean(),
            'avg_profit': wins['return'].mean() if len(wins) > 0 else 0,
            'avg_loss': abs(losses['return'].mean()) if len(losses) > 0 else 0,
            'optimal_holding': int(optimal_holding),
            'best_regime': 'any',  # Could be enhanced with regime detection
            'best_time': 'any',  # Could be enhanced with time analysis
            'confidence': min(len(results) / 100, 1)  # More occurrences = higher confidence
        }
    
    def calculate_feature_importance(self, cluster_features: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate importance of features for a cluster
        
        Args:
            cluster_features: Features of cluster members
            feature_names: Names of features
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        # Use variance as a simple importance measure
        variances = np.var(cluster_features, axis=0)
        total_var = np.sum(variances)
        
        if total_var == 0:
            return {name: 1.0 / len(feature_names) for name in feature_names[:cluster_features.shape[1]]}
        
        importance = variances / total_var
        
        return {
            name: float(importance[i]) 
            for i, name in enumerate(feature_names[:cluster_features.shape[1]])
        }
    
    def find_anomaly_patterns(self, features: np.ndarray, data: pd.DataFrame) -> List[Dict]:
        """
        Find anomaly patterns (rare but potentially profitable)
        
        Args:
            features: Feature matrix
            data: Market data
        
        Returns:
            List[Dict]: Anomaly patterns
        """
        self.logger.info("🔮 Finding anomaly patterns...")
        
        patterns = []
        
        try:
            # Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = iso_forest.fit_predict(features)
            
            # Analyze anomalies
            anomaly_indices = np.where(anomalies == -1)[0]
            
            if len(anomaly_indices) > 5:
                # Check profitability of anomalies
                profitability = self.analyze_cluster_profitability(
                    anomalies == -1, data
                )
                
                if profitability['win_rate'] > 0.6:  # High win rate for anomalies
                    pattern = {
                        'name': 'ML_Anomaly_Pattern',
                        'type': 'ml_anomaly',
                        'discovery_method': 'isolation_forest',
                        
                        'entry_conditions': {
                            'anomaly_score_threshold': -0.5,
                            'confirmation_required': True
                        },
                        
                        'exit_conditions': {
                            'holding_period': profitability['optimal_holding'],
                            'profit_target': profitability['avg_profit'] * 2,  # Higher targets for anomalies
                            'stop_loss': profitability['avg_loss']
                        },
                        
                        'statistics': {
                            'occurrences': len(anomaly_indices),
                            'preliminary_win_rate': profitability['win_rate'],
                            'avg_return': profitability['avg_return']
                        },
                        
                        'confidence': profitability['confidence'] * 0.8  # Lower confidence for anomalies
                    }
                    
                    patterns.append(pattern)
                    self.logger.info(f"  Found anomaly pattern (WR: {profitability['win_rate']:.1%})")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error finding anomalies: {e}")
            return []
    
    def mine_sequences(self, data: pd.DataFrame, min_length: int = 3) -> List[Dict]:
        """
        Mine repeating sequences in price action
        
        Args:
            data: Market data
            min_length: Minimum sequence length
        
        Returns:
            List[Dict]: Sequence patterns
        """
        self.logger.info("🔄 Mining repeating sequences...")
        
        patterns = []
        
        try:
            # Convert price moves to symbols
            returns = data['Close'].pct_change().fillna(0)
            
            # Discretize returns
            symbols = []
            for ret in returns:
                if ret > 0.002:
                    symbols.append('U')  # Up
                elif ret < -0.002:
                    symbols.append('D')  # Down
                else:
                    symbols.append('N')  # Neutral
            
            # Find repeating sequences
            sequences = self.find_repeating_sequences(symbols, min_length)
            
            for seq, occurrences in sequences.items():
                if occurrences < 10:  # Need minimum occurrences
                    continue
                
                # Analyze sequence profitability
                profitability = self.analyze_sequence_profitability(seq, symbols, data)
                
                if profitability['win_rate'] > 0.55:
                    pattern = {
                        'name': f'ML_Sequence_{"".join(seq)}',
                        'type': 'ml_sequence',
                        'discovery_method': 'sequence_mining',
                        
                        'entry_conditions': {
                            'sequence': seq,
                            'sequence_length': len(seq),
                            'match_threshold': 0.8
                        },
                        
                        'exit_conditions': {
                            'holding_period': profitability['optimal_holding'],
                            'profit_target': profitability['avg_profit'],
                            'stop_loss': profitability['avg_loss']
                        },
                        
                        'statistics': {
                            'occurrences': occurrences,
                            'preliminary_win_rate': profitability['win_rate'],
                            'avg_return': profitability['avg_return']
                        },
                        
                        'confidence': min(occurrences / 50, 1) * profitability['win_rate']
                    }
                    
                    patterns.append(pattern)
                    self.logger.info(f"  Found sequence pattern: {''.join(seq)} (Occ: {occurrences})")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error mining sequences: {e}")
            return []
    
    def find_repeating_sequences(self, symbols: List[str], min_length: int) -> Dict[Tuple, int]:
        """
        Find repeating sequences in symbol list
        
        Args:
            symbols: List of symbols
            min_length: Minimum sequence length
        
        Returns:
            Dict[Tuple, int]: Sequences and their occurrence counts
        """
        sequences = {}
        
        for length in range(min_length, min(10, len(symbols) // 10)):
            for i in range(len(symbols) - length):
                seq = tuple(symbols[i:i+length])
                
                if seq in sequences:
                    sequences[seq] += 1
                else:
                    sequences[seq] = 1
        
        # Filter for meaningful sequences
        return {seq: count for seq, count in sequences.items() 
                if count >= 5 and len(set(seq)) > 1}  # Not all same symbol
    
    def analyze_sequence_profitability(self, sequence: Tuple, symbols: List[str], 
                                      data: pd.DataFrame) -> Dict:
        """
        Analyze profitability after sequence occurs
        
        Args:
            sequence: Sequence pattern
            symbols: Full symbol list
            data: Market data
        
        Returns:
            Dict: Profitability metrics
        """
        results = []
        seq_len = len(sequence)
        
        # Find sequence occurrences
        for i in range(len(symbols) - seq_len - 10):
            if tuple(symbols[i:i+seq_len]) == sequence:
                # Check performance after sequence
                entry_idx = i + seq_len
                
                if entry_idx < len(data) - 10:
                    entry_price = data['Close'].iloc[entry_idx]
                    
                    for exit_offset in [1, 2, 3, 5, 10]:
                        if entry_idx + exit_offset < len(data):
                            exit_price = data['Close'].iloc[entry_idx + exit_offset]
                            return_pct = (exit_price - entry_price) / entry_price
                            
                            results.append({
                                'holding_period': exit_offset,
                                'return': return_pct,
                                'win': return_pct > 0
                            })
        
        if not results:
            return {
                'win_rate': 0,
                'avg_return': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'optimal_holding': 3
            }
        
        # Calculate metrics
        results_df = pd.DataFrame(results)
        
        # Best holding period
        holding_performance = results_df.groupby('holding_period').agg({
            'win': 'mean',
            'return': 'mean'
        })
        
        if not holding_performance.empty:
            optimal_holding = holding_performance['return'].idxmax()
            best_results = results_df[results_df['holding_period'] == optimal_holding]
            
            wins = best_results[best_results['win']]
            losses = best_results[~best_results['win']]
            
            return {
                'win_rate': best_results['win'].mean(),
                'avg_return': best_results['return'].mean(),
                'avg_profit': wins['return'].mean() if len(wins) > 0 else 0,
                'avg_loss': abs(losses['return'].mean()) if len(losses) > 0 else 0,
                'optimal_holding': int(optimal_holding)
            }
        else:
            return {
                'win_rate': 0,
                'avg_return': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'optimal_holding': 3
            }