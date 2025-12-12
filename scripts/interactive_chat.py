"""Interactive Chat for Harvey Multimodal RAG.

Allows querying the system for specific ZIP codes to see the retrieval and fusion process in action.
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force CPU to avoid OOM on busy nodes
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    console = Console()
except ImportError:
    class Console:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print("-" * 40)
        def status(self, msg): 
            print(f"... {msg} ...")
            return self
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

    console = Console()
    def Panel(x, title=None, border_style=None): return f"\n[{title}]\n{x}\n"
    def Markdown(x): return x

load_dotenv()

from app.core.dataio.storage import DataLocator
from app.core.retrieval.retriever import Retriever
from app.core.models.split_client import SplitPipelineClient
from app.core.retrieval.context_packager import package_context
from app.core.eval.ground_truth import FloodDepthGroundTruth
from app.core.dataio.schemas import RAGQuery

def main():
    parser = argparse.ArgumentParser(description="Harvey RAG Interactive Chat")
    parser.add_argument("--zip", type=str, required=True, help="ZIP code to query")
    parser.add_argument("--start", type=str, default="2017-08-25", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2017-09-02", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no_visual", action="store_true", help="Disable visual analysis")
    args = parser.parse_args()

    console.rule("[bold blue]Harvey Multimodal RAG System[/bold blue]")
    console.print(f"Query: ZIP [bold]{args.zip}[/bold] | {args.start} to {args.end}")

    # Initialize
    locator = DataLocator(Path("data"))
    retriever = Retriever(locator)
    client = SplitPipelineClient(enable_visual=not args.no_visual)
    
    # Ground Truth
    gt_loader = FloodDepthGroundTruth(Path("data/processed/flood_depth_by_zip.json"))
    gt = gt_loader.score(args.zip)

    # 1. Retrieval
    query = RAGQuery(zip=args.zip, start=args.start, end=args.end, k_tiles=6, n_text=20)
    with console.status("[bold green]Retrieving multimodal evidence...[/bold green]"):
        candidates_obj = retriever.retrieve(query)
        import dataclasses
        candidates = dataclasses.asdict(candidates_obj)
        context = package_context(candidates)
    
    # Show retrieval stats
    tweets = len(context.get("tweets", []))
    calls = len(context.get("calls", []))
    gauges = len(context.get("gauges", []))
    tiles = len(context.get("imagery_tiles", []))
    
    console.print(Panel(
        f"Tweets: {tweets}\n311 Calls: {calls}\nRain Sensors: {gauges}\nSatellite Tiles: {tiles}",
        title="Retrieval Stats",
        border_style="cyan"
    ))

    # 2. Fusion
    with console.status("[bold purple]Running Multimodal Fusion Engine...[/bold purple]"):
        response = client.infer(
            zip_code=args.zip,
            time_window={"start": args.start, "end": args.end},
            **context
        )
    
    # 3. Output
    estimates = response.get("estimates", {})
    impact_pct = estimates.get("flood_impact_pct", 0.0)
    reasoning = response.get("reasoning", "")
    
    # Display Result
    console.print(Panel(
        Markdown(reasoning),
        title=f"Reasoning (Confidence: {estimates.get('confidence', 'N/A')})",
        border_style="green"
    ))
    
    console.print(f"\n[bold]Predicted Flood Impact:[/bold] [magenta]{impact_pct}%[/magenta]")
    console.print(f"[bold]Roads Affected:[/bold] {', '.join(estimates.get('roads_impacted', []))}")
    
    console.rule("[bold red]Ground Truth Validation[/bold red]")
    console.print(f"Actual Flooded Area: [bold]{gt['flooded_pct']}%[/bold]")
    console.print(f"FEMA Claims: [bold]{gt['claim_count']:,}[/bold] claims")
    console.print(f"Total Paid: [bold]${gt['total_claim_amount']:,.2f}[/bold]")

    error = abs(impact_pct - gt['flooded_pct'])
    console.print(f"\nError (MAE): [bold red]{error:.2f}%[/bold red]")

if __name__ == "__main__":
    main()
