"""
Command line interface for QuLab.

Provides comprehensive CLI commands for running teleportation experiments,
analyzing results, managing governance, and performing encoding operations.
"""

import typer
from typing import Optional, List
import json
import asyncio
from pathlib import Path
import logging

from .quantum.teleportation import TeleportationProtocol
from .governance.ledger import EvidenceLedger
from .governance.forecasting import MonteCarloForecaster
from .governance.cadence import CadencePlanner
from .encoding.base_n import BaseNEncoder, BaseNDecoder
from .encoding.packing import PackingOptimizer
from .encoding.efficiency import EfficiencyAnalyzer
from .io.results import ResultsManager
from .io.plots import PlotManager, PlotConfig
from .api.teleport import router as teleport_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="QuLab: Quantum Laboratory Framework for Teleportation Research")

# Create subcommands
teleport_app = typer.Typer(help="Quantum teleportation operations")
governance_app = typer.Typer(help="Governance and evidence management")
encoding_app = typer.Typer(help="Base-N encoding operations")
analysis_app = typer.Typer(help="Analysis and visualization")
serve_app = typer.Typer(help="API server operations")

app.add_typer(teleport_app, name="teleport")
app.add_typer(governance_app, name="governance")
app.add_typer(encoding_app, name="encode")
app.add_typer(analysis_app, name="analyze")
app.add_typer(serve_app, name="serve")


@teleport_app.command("run")
def teleport_run(
    alpha: float = typer.Option(0.6, help="Amplitude of |0⟩ state"),
    beta: float = typer.Option(0.8, help="Amplitude of |1⟩ state"),
    shots: int = typer.Option(1024, help="Number of measurement shots"),
    output: Optional[str] = typer.Option(None, help="Output file path"),
    save_results: bool = typer.Option(True, help="Save results to storage")
):
    """Run quantum teleportation experiment."""
    try:
        # Validate normalization
        import numpy as np
        if not np.isclose(abs(alpha)**2 + abs(beta)**2, 1.0, atol=1e-10):
            typer.echo("Error: Quantum state must be normalized (|α|² + |β|² = 1)", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"Running teleportation experiment...")
        typer.echo(f"State: |ψ⟩ = {alpha:.3f}|0⟩ + {beta:.3f}|1⟩")
        typer.echo(f"Shots: {shots}")
        
        # Create protocol and run experiment
        protocol = TeleportationProtocol()
        result = protocol.teleport(alpha, beta, shots)
        
        # Display results
        typer.echo(f"\nResults:")
        typer.echo(f"  Fidelity: {result.fidelity:.6f}")
        typer.echo(f"  Success Probability: {result.success_probability:.6f}")
        typer.echo(f"  Execution Time: {result.execution_time:.3f} seconds")
        typer.echo(f"  Classical Bits: {result.classical_bits}")
        
        # Save results if requested
        if save_results:
            results_manager = ResultsManager()
            from .io.schemas import TeleportationSchema
            import uuid
            
            schema = TeleportationSchema(
                experiment_id=str(uuid.uuid4()),
                timestamp=result.execution_time,
                alpha=alpha,
                beta=beta,
                fidelity=result.fidelity,
                success_probability=result.success_probability,
                shots=result.shots,
                execution_time=result.execution_time,
                measurement_results=result.measurement_results,
                classical_bits=list(result.classical_bits),
                metadata={"cli_run": True}
            )
            
            filepath = results_manager.save_teleportation_result(schema)
            typer.echo(f"  Results saved to: {filepath}")
        
        # Save to output file if specified
        if output:
            output_data = {
                "alpha": alpha,
                "beta": beta,
                "shots": shots,
                "fidelity": result.fidelity,
                "success_probability": result.success_probability,
                "execution_time": result.execution_time,
                "measurement_results": result.measurement_results,
                "classical_bits": list(result.classical_bits)
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            typer.echo(f"  Results exported to: {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@teleport_app.command("bands")
def teleport_bands(
    alpha: float = typer.Option(0.6, help="Amplitude of |0⟩ state"),
    beta: float = typer.Option(0.8, help="Amplitude of |1⟩ state"),
    shots: int = typer.Option(1024, help="Number of shots per trial"),
    trials: int = typer.Option(100, help="Number of Monte Carlo trials"),
    output: Optional[str] = typer.Option(None, help="Output file path")
):
    """Analyze fidelity confidence bands using Monte Carlo sampling."""
    try:
        # Validate normalization
        import numpy as np
        if not np.isclose(abs(alpha)**2 + abs(beta)**2, 1.0, atol=1e-10):
            typer.echo("Error: Quantum state must be normalized (|α|² + |β|² = 1)", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"Analyzing fidelity bands...")
        typer.echo(f"State: |ψ⟩ = {alpha:.3f}|0⟩ + {beta:.3f}|1⟩")
        typer.echo(f"Shots per trial: {shots}")
        typer.echo(f"Number of trials: {trials}")
        
        # Create protocol and analyze
        protocol = TeleportationProtocol()
        bands = protocol.analyze_fidelity_bands(alpha, beta, shots, trials)
        
        # Display results
        typer.echo(f"\nFidelity Bands:")
        typer.echo(f"  Mean: {bands['mean']:.6f}")
        typer.echo(f"  Std: {bands['std']:.6f}")
        typer.echo(f"  95% CI: [{bands['confidence_95_lower']:.6f}, {bands['confidence_95_upper']:.6f}]")
        typer.echo(f"  99% CI: [{bands['confidence_99_lower']:.6f}, {bands['confidence_99_upper']:.6f}]")
        
        # Save to output file if specified
        if output:
            output_data = {
                "alpha": alpha,
                "beta": beta,
                "shots": shots,
                "trials": trials,
                "bands": bands
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            typer.echo(f"  Results exported to: {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@governance_app.command("init")
def governance_init(
    alpha_prior: float = typer.Option(1.0, help="Prior alpha parameter"),
    beta_prior: float = typer.Option(1.0, help="Prior beta parameter"),
    output: Optional[str] = typer.Option(None, help="Output file path")
):
    """Initialize evidence ledger."""
    try:
        typer.echo("Initializing evidence ledger...")
        
        ledger = EvidenceLedger(alpha_prior, beta_prior)
        
        typer.echo(f"Evidence ledger initialized:")
        typer.echo(f"  Alpha prior: {alpha_prior}")
        typer.echo(f"  Beta prior: {beta_prior}")
        typer.echo(f"  Current mean: {ledger.get_mean():.6f}")
        typer.echo(f"  Current std: {ledger.get_std():.6f}")
        
        # Save ledger if output specified
        if output:
            ledger_data = {
                "alpha_prior": alpha_prior,
                "beta_prior": beta_prior,
                "alpha": ledger.alpha,
                "beta": ledger.beta,
                "evidence_entries": ledger.export_evidence()
            }
            
            with open(output, 'w') as f:
                json.dump(ledger_data, f, indent=2)
            typer.echo(f"  Ledger saved to: {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@governance_app.command("update")
def governance_update(
    fidelity: float = typer.Option(..., help="Observed fidelity"),
    confidence: float = typer.Option(0.95, help="Confidence in measurement"),
    shots: int = typer.Option(1000, help="Number of measurement shots"),
    experiment_id: str = typer.Option(..., help="Experiment identifier"),
    ledger_file: Optional[str] = typer.Option(None, help="Ledger file path"),
    output: Optional[str] = typer.Option(None, help="Output file path")
):
    """Update evidence ledger with new measurement."""
    try:
        # Load or create ledger
        if ledger_file and Path(ledger_file).exists():
            with open(ledger_file, 'r') as f:
                ledger_data = json.load(f)
            
            ledger = EvidenceLedger(ledger_data["alpha_prior"], ledger_data["beta_prior"])
            ledger.alpha = ledger_data["alpha"]
            ledger.beta = ledger_data["beta"]
            ledger.import_evidence(ledger_data["evidence_entries"])
        else:
            ledger = EvidenceLedger()
        
        typer.echo("Updating evidence ledger...")
        typer.echo(f"  Fidelity: {fidelity}")
        typer.echo(f"  Confidence: {confidence}")
        typer.echo(f"  Shots: {shots}")
        typer.echo(f"  Experiment ID: {experiment_id}")
        
        # Update ledger
        ledger.update_evidence(fidelity, confidence, shots, experiment_id)
        
        # Display updated statistics
        typer.echo(f"\nUpdated Statistics:")
        typer.echo(f"  Mean fidelity: {ledger.get_mean():.6f}")
        typer.echo(f"  Std fidelity: {ledger.get_std():.6f}")
        typer.echo(f"  Effective sample size: {ledger.get_effective_sample_size():.1f}")
        
        ci_95 = ledger.get_credible_interval(0.95)
        typer.echo(f"  95% Credible Interval: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")
        
        # Save updated ledger
        output_file = output or ledger_file or "ledger.json"
        ledger_data = {
            "alpha_prior": ledger.alpha_prior,
            "beta_prior": ledger.beta_prior,
            "alpha": ledger.alpha,
            "beta": ledger.beta,
            "evidence_entries": ledger.export_evidence()
        }
        
        with open(output_file, 'w') as f:
            json.dump(ledger_data, f, indent=2)
        typer.echo(f"  Ledger saved to: {output_file}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@governance_app.command("forecast")
def governance_forecast(
    horizon: int = typer.Option(30, help="Forecast horizon in days"),
    samples: int = typer.Option(10000, help="Number of Monte Carlo samples"),
    ledger_file: Optional[str] = typer.Option(None, help="Ledger file path"),
    output: Optional[str] = typer.Option(None, help="Output file path")
):
    """Generate fidelity forecasts using Monte Carlo methods."""
    try:
        # Load ledger
        if ledger_file and Path(ledger_file).exists():
            with open(ledger_file, 'r') as f:
                ledger_data = json.load(f)
            
            ledger = EvidenceLedger(ledger_data["alpha_prior"], ledger_data["beta_prior"])
            ledger.alpha = ledger_data["alpha"]
            ledger.beta = ledger_data["beta"]
            ledger.import_evidence(ledger_data["evidence_entries"])
        else:
            typer.echo("Error: No ledger file provided or file not found", err=True)
            raise typer.Exit(1)
        
        typer.echo("Generating fidelity forecasts...")
        typer.echo(f"  Horizon: {horizon} days")
        typer.echo(f"  Monte Carlo samples: {samples}")
        
        # Create forecaster and generate forecast
        forecaster = MonteCarloForecaster(ledger)
        forecast = forecaster.forecast_fidelity(horizon, samples)
        
        # Display results
        typer.echo(f"\nForecast Results:")
        typer.echo(f"  Current mean: {ledger.get_mean():.6f}")
        typer.echo(f"  Current std: {ledger.get_std():.6f}")
        typer.echo(f"  Forecast mean (final): {forecast.mean_forecast[-1]:.6f}")
        typer.echo(f"  Forecast std (final): {forecast.std_forecast[-1]:.6f}")
        
        # Save forecast if output specified
        if output:
            forecast_data = {
                "horizon": horizon,
                "samples": samples,
                "current_mean": ledger.get_mean(),
                "current_std": ledger.get_std(),
                "forecast": {
                    "mean_forecast": forecast.mean_forecast,
                    "std_forecast": forecast.std_forecast,
                    "confidence_intervals": forecast.confidence_intervals,
                    "forecast_dates": [d.isoformat() for d in forecast.forecast_dates]
                }
            }
            
            with open(output, 'w') as f:
                json.dump(forecast_data, f, indent=2)
            typer.echo(f"  Forecast saved to: {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@encoding_app.command("encode")
def encoding_encode(
    data: str = typer.Option(..., help="Data to encode"),
    base: int = typer.Option(64, help="Encoding base"),
    alphabet: Optional[str] = typer.Option(None, help="Custom alphabet"),
    error_detection: bool = typer.Option(False, help="Enable error detection"),
    output: Optional[str] = typer.Option(None, help="Output file path")
):
    """Encode data using base-N encoding."""
    try:
        typer.echo(f"Encoding data with base-{base}...")
        typer.echo(f"  Input data: {data[:50]}{'...' if len(data) > 50 else ''}")
        typer.echo(f"  Base: {base}")
        typer.echo(f"  Error detection: {error_detection}")
        
        # Create encoder
        encoder = BaseNEncoder(base, alphabet, error_detection)
        result = encoder.encode(data)
        
        # Display results
        typer.echo(f"\nEncoding Results:")
        typer.echo(f"  Original length: {result.original_length} bytes")
        typer.echo(f"  Encoded length: {result.encoded_length} characters")
        typer.echo(f"  Compression ratio: {result.compression_ratio:.3f}")
        typer.echo(f"  Padding: {result.padding} bytes")
        typer.echo(f"  Encoded data: {result.encoded_data[:100]}{'...' if len(result.encoded_data) > 100 else ''}")
        
        # Save results if output specified
        if output:
            output_data = {
                "original_data": data,
                "base": base,
                "alphabet": result.alphabet,
                "encoded_data": result.encoded_data,
                "original_length": result.original_length,
                "encoded_length": result.encoded_length,
                "compression_ratio": result.compression_ratio,
                "padding": result.padding,
                "error_detection": error_detection
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            typer.echo(f"  Results saved to: {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@encoding_app.command("decode")
def encoding_decode(
    encoded_data: str = typer.Option(..., help="Encoded data to decode"),
    base: int = typer.Option(64, help="Decoding base"),
    alphabet: Optional[str] = typer.Option(None, help="Custom alphabet"),
    error_detection: bool = typer.Option(False, help="Error detection enabled"),
    output: Optional[str] = typer.Option(None, help="Output file path")
):
    """Decode base-N encoded data."""
    try:
        typer.echo(f"Decoding base-{base} data...")
        typer.echo(f"  Encoded data: {encoded_data[:50]}{'...' if len(encoded_data) > 50 else ''}")
        typer.echo(f"  Base: {base}")
        typer.echo(f"  Error detection: {error_detection}")
        
        # Create decoder
        decoder = BaseNDecoder(base, alphabet, error_detection)
        decoded_data = decoder.decode_to_string(encoded_data)
        
        # Display results
        typer.echo(f"\nDecoding Results:")
        typer.echo(f"  Decoded data: {decoded_data[:100]}{'...' if len(decoded_data) > 100 else ''}")
        typer.echo(f"  Decoded length: {len(decoded_data)} characters")
        
        # Save results if output specified
        if output:
            output_data = {
                "encoded_data": encoded_data,
                "base": base,
                "alphabet": decoder.alphabet,
                "decoded_data": decoded_data,
                "decoded_length": len(decoded_data),
                "error_detection": error_detection
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            typer.echo(f"  Results saved to: {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@encoding_app.command("optimize")
def encoding_optimize(
    data: str = typer.Option(..., help="Data to optimize encoding for"),
    bases: Optional[str] = typer.Option(None, help="Comma-separated list of bases to test"),
    output: Optional[str] = typer.Option(None, help="Output file path")
):
    """Optimize encoding parameters for given data."""
    try:
        typer.echo("Optimizing encoding parameters...")
        typer.echo(f"  Input data: {data[:50]}{'...' if len(data) > 50 else ''}")
        
        # Parse bases if provided
        test_bases = None
        if bases:
            test_bases = [int(b.strip()) for b in bases.split(',')]
        
        # Create optimizer
        optimizer = PackingOptimizer(test_bases)
        result = optimizer.optimize_packing(data)
        
        # Display results
        typer.echo(f"\nOptimization Results:")
        typer.echo(f"  Optimal base: {result.optimal_base}")
        typer.echo(f"  Compression ratio: {result.compression_ratio:.3f}")
        typer.echo(f"  Packing efficiency: {result.packing_efficiency:.3f}")
        typer.echo(f"  Data type: {result.data_type}")
        typer.echo(f"  Optimal alphabet: {result.optimal_alphabet[:50]}{'...' if len(result.optimal_alphabet) > 50 else ''}")
        
        typer.echo(f"\nRecommendations:")
        for rec in result.recommendations:
            typer.echo(f"  - {rec}")
        
        # Save results if output specified
        if output:
            output_data = {
                "input_data": data,
                "optimal_base": result.optimal_base,
                "optimal_alphabet": result.optimal_alphabet,
                "compression_ratio": result.compression_ratio,
                "packing_efficiency": result.packing_efficiency,
                "data_type": result.data_type,
                "recommendations": result.recommendations
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            typer.echo(f"  Results saved to: {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@analysis_app.command("plot")
def analysis_plot(
    data_type: str = typer.Option(..., help="Type of data to plot (teleportation, governance, encoding)"),
    interactive: bool = typer.Option(True, help="Create interactive plots"),
    output_dir: str = typer.Option("plots", help="Output directory for plots")
):
    """Generate plots from experimental data."""
    try:
        typer.echo(f"Generating {data_type} plots...")
        typer.echo(f"  Interactive: {interactive}")
        typer.echo(f"  Output directory: {output_dir}")
        
        # Create plot manager
        results_manager = ResultsManager()
        plot_manager = PlotManager(results_manager, output_dir)
        
        # Create plot configuration
        config = PlotConfig(interactive=interactive)
        
        # Load data and create plots
        if data_type == "teleportation":
            results = results_manager.load_teleportation_results()
            if not results:
                typer.echo("No teleportation results found", err=True)
                raise typer.Exit(1)
            
            plot_path = plot_manager.plot_teleportation_fidelity(results, config)
            
        elif data_type == "governance":
            results = results_manager.load_governance_results()
            if not results:
                typer.echo("No governance results found", err=True)
                raise typer.Exit(1)
            
            plot_path = plot_manager.plot_governance_evidence(results, config)
            
        elif data_type == "encoding":
            results = results_manager.load_encoding_results()
            if not results:
                typer.echo("No encoding results found", err=True)
                raise typer.Exit(1)
            
            plot_path = plot_manager.plot_encoding_efficiency(results, config)
            
        else:
            typer.echo(f"Unknown data type: {data_type}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"  Plot saved to: {plot_path}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@analysis_app.command("stats")
def analysis_stats(
    data_type: str = typer.Option(..., help="Type of data to analyze (teleportation, governance, encoding)")
):
    """Display statistics for experimental data."""
    try:
        typer.echo(f"Analyzing {data_type} statistics...")
        
        # Create results manager
        results_manager = ResultsManager()
        stats = results_manager.get_statistics(data_type)
        
        if not stats:
            typer.echo(f"No {data_type} data found")
            return
        
        # Display statistics
        typer.echo(f"\n{data_type.title()} Statistics:")
        typer.echo(f"  Total count: {stats.get('total_count', 0)}")
        
        if 'date_range' in stats:
            date_range = stats['date_range']
            if date_range['start']:
                typer.echo(f"  Date range: {date_range['start']} to {date_range['end']}")
        
        # Type-specific statistics
        if data_type == "teleportation" and 'fidelity' in stats:
            fidelity_stats = stats['fidelity']
            typer.echo(f"  Fidelity:")
            typer.echo(f"    Mean: {fidelity_stats['mean']:.6f}")
            typer.echo(f"    Std: {fidelity_stats['std']:.6f}")
            typer.echo(f"    Min: {fidelity_stats['min']:.6f}")
            typer.echo(f"    Max: {fidelity_stats['max']:.6f}")
            typer.echo(f"    Median: {fidelity_stats['median']:.6f}")
        
        elif data_type == "encoding" and 'compression' in stats:
            compression_stats = stats['compression']
            typer.echo(f"  Compression:")
            typer.echo(f"    Mean ratio: {compression_stats['mean_ratio']:.3f}")
            typer.echo(f"    Best ratio: {compression_stats['best_ratio']:.3f}")
            typer.echo(f"    Worst ratio: {compression_stats['worst_ratio']:.3f}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@serve_app.command("start")
def serve_start(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload")
):
    """Start the QuLab API server."""
    try:
        typer.echo(f"Starting QuLab API server...")
        typer.echo(f"  Host: {host}")
        typer.echo(f"  Port: {port}")
        typer.echo(f"  Reload: {reload}")
        
        import uvicorn
        from fastapi import FastAPI
        
        # Create FastAPI app
        app = FastAPI(
            title="QuLab API",
            description="Quantum Laboratory Framework for Teleportation Research",
            version="0.1.0"
        )
        
        # Include routers
        app.include_router(teleport_router)
        # Add other routers as they're implemented
        
        # Start server
        uvicorn.run(app, host=host, port=port, reload=reload)
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
