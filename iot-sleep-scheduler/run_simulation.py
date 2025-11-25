#!/usr/bin/env python3
"""
Quick Simulation Runner
Convenient script to run the complete IoT sleep scheduling simulation
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from main import SimulationRunner
from visualization.generate_graphs import GraphGenerator
from utils.summary_generator import SummaryGenerator


def main():
    """Run complete simulation with all outputs"""
    print("="*60)
    print("IoT SLEEP SCHEDULING SIMULATION")
    print("Energy-Efficient Power Optimization for IoT Sensor Nodes")
    print("="*60)

    # Create simulation runner
    runner = SimulationRunner()

    # Run basic simulation
    print("\n🚀 Starting IoT Sleep Scheduling Simulation...")
    print("   Running basic simulation with 2 nodes for 24 hours")

    try:
        results = runner.run_basic_simulation(duration_hours=24.0, node_count=2)

        # Save results
        results_path = runner.save_results(results, "simulation_results.json")
        print(f"✓ Simulation completed successfully!")
        print(f"✓ Results saved to: {results_path}")

        # Generate graphs
        print("\n📊 Generating visualization graphs...")
        graph_generator = GraphGenerator()
        graphs_path = graph_generator.generate_all_graphs(results)
        print(f"✓ Graphs generated in: {graphs_path}")

        # Generate summary
        print("\n📋 Generating summary metrics...")
        summary_generator = SummaryGenerator()
        summary_path = summary_generator.generate_summary(results)
        print(f"✓ Summary generated: {summary_path}")

        # Print key results
        print("\n" + "="*60)
        print("SIMULATION RESULTS SUMMARY")
        print("="*60)

        if 'overall_metrics' in results:
            metrics = results['overall_metrics']
            print(f"📈 Total Simulation Time: 24.0 hours")
            print(f"🔋 Total Energy Consumed: {metrics.get('total_energy_consumed_all_nodes', 0):.2f} mAh")
            print(f"⚡ Total Wake-ups: {metrics.get('total_wakeups_all_nodes', 0)}")
            print(f"📡 Total Transmissions: {metrics.get('total_transmissions_all_nodes', 0)}")
            print(f"🔋 Average Battery Level: {metrics.get('average_battery_percentage', 0):.1f}%")
            print(f"💀 Nodes Depleted: {metrics.get('nodes_depleted', 0)}")

            if 'node_results' in results:
                for node_id, node_data in results['node_results'].items():
                    scheduler_type = "Static" if "static" in node_id else "Adaptive"
                    print(f"   {node_id} ({scheduler_type}): {node_data['final_battery_percentage']:.1f}% battery")

        print(f"\n📁 Output Directory: {Path(results_path).parent}")
        print("📄 Generated Files:")
        print("   - simulation_log.csv (Raw simulation data)")
        print("   - simulation_results.json (Complete results)")
        print("   - summary_metrics.json (Performance summary)")
        print("   - graphs/*.png (Visualization graphs)")

        print("\n✨ Simulation completed successfully! ✨")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()