"""
Plotting utilities for QuLab results.

Provides comprehensive plotting capabilities for teleportation results,
governance data, and encoding efficiency with interactive and static plots.
"""

from typing import List, Dict, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from .schemas import TeleportationSchema, GovernanceSchema, EncodingSchema
from .results import ResultsManager

logger = logging.getLogger(__name__)


class PlotConfig(BaseModel):
    """Configuration for plotting."""
    
    width: int = Field(800, description="Plot width in pixels")
    height: int = Field(600, description="Plot height in pixels")
    title: str = Field("", description="Plot title")
    xlabel: str = Field("", description="X-axis label")
    ylabel: str = Field("", description="Y-axis label")
    theme: str = Field("plotly_white", description="Plot theme")
    color_scheme: str = Field("viridis", description="Color scheme")
    show_legend: bool = Field(True, description="Show legend")
    interactive: bool = Field(True, description="Use interactive plots")
    
    class Config:
        arbitrary_types_allowed = True


class PlotManager:
    """
    Manager for creating and saving plots of QuLab results.
    
    Supports both static (matplotlib) and interactive (plotly) plots
    with comprehensive customization options.
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None,
                 output_path: Union[str, Path] = "plots"):
        """
        Initialize plot manager.
        
        Args:
            results_manager: Results manager for loading data
            output_path: Directory for saving plots
        """
        self.results_manager = results_manager
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def plot_teleportation_fidelity(self, results: List[TeleportationSchema],
                                  config: Optional[PlotConfig] = None) -> str:
        """
        Plot teleportation fidelity over time.
        
        Args:
            results: List of teleportation results
            config: Plot configuration
            
        Returns:
            Path to saved plot
        """
        if not results:
            raise ValueError("No teleportation results provided")
        
        config = config or PlotConfig()
        
        # Prepare data
        df = pd.DataFrame([result.dict() for result in results])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if config.interactive:
            return self._plot_teleportation_fidelity_interactive(df, config)
        else:
            return self._plot_teleportation_fidelity_static(df, config)
    
    def _plot_teleportation_fidelity_interactive(self, df: pd.DataFrame, 
                                               config: PlotConfig) -> str:
        """Create interactive fidelity plot."""
        fig = go.Figure()
        
        # Add fidelity line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['fidelity'],
            mode='lines+markers',
            name='Fidelity',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add confidence bands if available
        if 'success_probability' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['success_probability'],
                mode='lines',
                name='Success Probability',
                line=dict(color='green', width=1, dash='dash'),
                opacity=0.7
            ))
        
        # Update layout
        fig.update_layout(
            title=config.title or "Teleportation Fidelity Over Time",
            xaxis_title=config.xlabel or "Time",
            yaxis_title=config.ylabel or "Fidelity",
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=config.show_legend
        )
        
        # Format x-axis for dates
        fig.update_xaxes(
            tickformat="%Y-%m-%d %H:%M",
            tickangle=45
        )
        
        # Save plot
        filename = f"teleportation_fidelity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _plot_teleportation_fidelity_static(self, df: pd.DataFrame, 
                                          config: PlotConfig) -> str:
        """Create static fidelity plot."""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
        
        # Plot fidelity
        ax.plot(df['timestamp'], df['fidelity'], 'b-', linewidth=2, 
                marker='o', markersize=4, label='Fidelity')
        
        # Add success probability if available
        if 'success_probability' in df.columns:
            ax.plot(df['timestamp'], df['success_probability'], 'g--', 
                   linewidth=1, alpha=0.7, label='Success Probability')
        
        # Formatting
        ax.set_title(config.title or "Teleportation Fidelity Over Time")
        ax.set_xlabel(config.xlabel or "Time")
        ax.set_ylabel(config.ylabel or "Fidelity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Save plot
        filename = f"teleportation_fidelity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_governance_evidence(self, results: List[GovernanceSchema],
                               config: Optional[PlotConfig] = None) -> str:
        """
        Plot governance evidence accumulation.
        
        Args:
            results: List of governance results
            config: Plot configuration
            
        Returns:
            Path to saved plot
        """
        if not results:
            raise ValueError("No governance results provided")
        
        config = config or PlotConfig()
        
        # Prepare data
        df = pd.DataFrame([result.dict() for result in results])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if config.interactive:
            return self._plot_governance_evidence_interactive(df, config)
        else:
            return self._plot_governance_evidence_static(df, config)
    
    def _plot_governance_evidence_interactive(self, df: pd.DataFrame, 
                                            config: PlotConfig) -> str:
        """Create interactive governance evidence plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Mean Fidelity', 'Credible Intervals'),
            vertical_spacing=0.1
        )
        
        # Plot mean fidelity
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['mean_fidelity'],
                mode='lines+markers',
                name='Mean Fidelity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Plot credible intervals
        if 'credible_interval_95' in df.columns:
            # Extract lower and upper bounds
            lower_bounds = [ci[0] for ci in df['credible_interval_95']]
            upper_bounds = [ci[1] for ci in df['credible_interval_95']]
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=upper_bounds,
                    mode='lines',
                    name='95% CI Upper',
                    line=dict(color='red', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=lower_bounds,
                    mode='lines',
                    name='95% CI Lower',
                    line=dict(color='red', width=1, dash='dash'),
                    opacity=0.7,
                    fill='tonexty'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=config.title or "Governance Evidence Accumulation",
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=config.show_legend
        )
        
        # Format x-axis
        fig.update_xaxes(tickformat="%Y-%m-%d %H:%M", tickangle=45)
        
        # Save plot
        filename = f"governance_evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _plot_governance_evidence_static(self, df: pd.DataFrame, 
                                       config: PlotConfig) -> str:
        """Create static governance evidence plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(config.width/100, config.height/100))
        
        # Plot mean fidelity
        ax1.plot(df['timestamp'], df['mean_fidelity'], 'b-', linewidth=2, 
                marker='o', markersize=4, label='Mean Fidelity')
        ax1.set_title('Mean Fidelity')
        ax1.set_ylabel('Fidelity')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot credible intervals
        if 'credible_interval_95' in df.columns:
            lower_bounds = [ci[0] for ci in df['credible_interval_95']]
            upper_bounds = [ci[1] for ci in df['credible_interval_95']]
            
            ax2.fill_between(df['timestamp'], lower_bounds, upper_bounds, 
                           alpha=0.3, color='red', label='95% Credible Interval')
            ax2.plot(df['timestamp'], df['mean_fidelity'], 'b-', linewidth=2, 
                    label='Mean Fidelity')
        
        ax2.set_title('Credible Intervals')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fidelity')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        plt.xticks(rotation=45)
        
        # Save plot
        filename = f"governance_evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_encoding_efficiency(self, results: List[EncodingSchema],
                               config: Optional[PlotConfig] = None) -> str:
        """
        Plot encoding efficiency analysis.
        
        Args:
            results: List of encoding results
            config: Plot configuration
            
        Returns:
            Path to saved plot
        """
        if not results:
            raise ValueError("No encoding results provided")
        
        config = config or PlotConfig()
        
        # Prepare data
        df = pd.DataFrame([result.dict() for result in results])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if config.interactive:
            return self._plot_encoding_efficiency_interactive(df, config)
        else:
            return self._plot_encoding_efficiency_static(df, config)
    
    def _plot_encoding_efficiency_interactive(self, df: pd.DataFrame, 
                                            config: PlotConfig) -> str:
        """Create interactive encoding efficiency plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Compression Ratio by Base', 'Encoding Time',
                          'Alphabet Usage', 'Efficiency Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Compression ratio by base
        fig.add_trace(
            go.Bar(
                x=df['base'],
                y=df['compression_ratio'],
                name='Compression Ratio',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # Encoding time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df.get('encoding_time', [0] * len(df)),
                mode='lines+markers',
                name='Encoding Time',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        # Alphabet usage (if available)
        if 'efficiency_metrics' in df.columns:
            # Extract alphabet efficiency from metrics
            alphabet_eff = []
            for metrics in df['efficiency_metrics']:
                if isinstance(metrics, dict) and 'alphabet_efficiency' in metrics:
                    alphabet_eff.append(metrics['alphabet_efficiency'])
                else:
                    alphabet_eff.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=df['base'],
                    y=alphabet_eff,
                    name='Alphabet Efficiency',
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # Overall efficiency
        if 'efficiency_metrics' in df.columns:
            overall_eff = []
            for metrics in df['efficiency_metrics']:
                if isinstance(metrics, dict) and 'overall_efficiency' in metrics:
                    overall_eff.append(metrics['overall_efficiency'])
                else:
                    overall_eff.append(0)
            
            fig.add_trace(
                go.Scatter(
                    x=df['base'],
                    y=overall_eff,
                    mode='lines+markers',
                    name='Overall Efficiency',
                    line=dict(color='red')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=config.title or "Encoding Efficiency Analysis",
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=config.show_legend
        )
        
        # Save plot
        filename = f"encoding_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _plot_encoding_efficiency_static(self, df: pd.DataFrame, 
                                       config: PlotConfig) -> str:
        """Create static encoding efficiency plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                    figsize=(config.width/100, config.height/100))
        
        # Compression ratio by base
        ax1.bar(df['base'], df['compression_ratio'], color='blue', alpha=0.7)
        ax1.set_title('Compression Ratio by Base')
        ax1.set_xlabel('Base')
        ax1.set_ylabel('Compression Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Encoding time
        ax2.plot(df['timestamp'], df.get('encoding_time', [0] * len(df)), 
                'g-', linewidth=2, marker='o')
        ax2.set_title('Encoding Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # Alphabet efficiency
        if 'efficiency_metrics' in df.columns:
            alphabet_eff = []
            for metrics in df['efficiency_metrics']:
                if isinstance(metrics, dict) and 'alphabet_efficiency' in metrics:
                    alphabet_eff.append(metrics['alphabet_efficiency'])
                else:
                    alphabet_eff.append(0)
            
            ax3.bar(df['base'], alphabet_eff, color='orange', alpha=0.7)
            ax3.set_title('Alphabet Efficiency')
            ax3.set_xlabel('Base')
            ax3.set_ylabel('Efficiency')
            ax3.grid(True, alpha=0.3)
        
        # Overall efficiency
        if 'efficiency_metrics' in df.columns:
            overall_eff = []
            for metrics in df['efficiency_metrics']:
                if isinstance(metrics, dict) and 'overall_efficiency' in metrics:
                    overall_eff.append(metrics['overall_efficiency'])
                else:
                    overall_eff.append(0)
            
            ax4.plot(df['base'], overall_eff, 'r-', linewidth=2, marker='s')
            ax4.set_title('Overall Efficiency')
            ax4.set_xlabel('Base')
            ax4.set_ylabel('Efficiency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"encoding_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_comparison(self, results_dict: Dict[str, List[Any]], 
                       metric: str, config: Optional[PlotConfig] = None) -> str:
        """
        Plot comparison of different result types.
        
        Args:
            results_dict: Dictionary mapping names to result lists
            metric: Metric to compare
            config: Plot configuration
            
        Returns:
            Path to saved plot
        """
        config = config or PlotConfig()
        
        if config.interactive:
            return self._plot_comparison_interactive(results_dict, metric, config)
        else:
            return self._plot_comparison_static(results_dict, metric, config)
    
    def _plot_comparison_interactive(self, results_dict: Dict[str, List[Any]], 
                                   metric: str, config: PlotConfig) -> str:
        """Create interactive comparison plot."""
        fig = go.Figure()
        
        for name, results in results_dict.items():
            if not results:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame([result.dict() for result in results])
            
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.get('timestamp', range(len(df))),
                    y=df[metric],
                    mode='lines+markers',
                    name=name,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=config.title or f"Comparison: {metric}",
            xaxis_title=config.xlabel or "Time",
            yaxis_title=config.ylabel or metric,
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=config.show_legend
        )
        
        # Save plot
        filename = f"comparison_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _plot_comparison_static(self, results_dict: Dict[str, List[Any]], 
                              metric: str, config: PlotConfig) -> str:
        """Create static comparison plot."""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
        
        for name, results in results_dict.items():
            if not results:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame([result.dict() for result in results])
            
            if metric in df.columns:
                ax.plot(df.get('timestamp', range(len(df))), df[metric], 
                       linewidth=2, marker='o', label=name)
        
        ax.set_title(config.title or f"Comparison: {metric}")
        ax.set_xlabel(config.xlabel or "Time")
        ax.set_ylabel(config.ylabel or metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save plot
        filename = f"comparison_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
