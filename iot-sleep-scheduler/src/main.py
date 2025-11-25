"""
Main Simulation Runner for IoT Sleep Scheduling System
Complete entry point for running simulations with different configurations and generating results
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.node import NodeType
from simulation.simulator import Simulator
from simulation.logger import create_logger, LoggingContext
from config import config


class SimulationRunner:
    """
    Main simulation runner with comprehensive configuration and result generation

    Features:
    - Multiple simulation scenarios
    - Configurable parameters
    - Automated result generation
    - Performance benchmarking
    - Comparative analysis
    - Output file management
    """

    def __init__(self):
        """Initialize simulation runner"""
        self.results = {}
        self.simulation_configs = {}
        self.output_base_path = Path(config.simulation.GRAPHS_OUTPUT_DIR).parent
        self.output_base_path.mkdir(parents=True, exist_ok=True)

    def run_basic_simulation(self, duration_hours: float = 24.0, node_count: int = 2) -> Dict[str, Any]:
        """
        Run basic simulation with default configuration

        Args:
            duration_hours: Simulation duration in hours
            node_count: Number of nodes to simulate

        Returns:
            Simulation results
        """
        print(f"Running basic simulation: {duration_hours}h, {node_count} nodes")

        # Create logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_base_path / "logs" / f"basic_simulation_{timestamp}.csv"

        with LoggingContext(log_file_path=log_file) as logger:
            # Create simulator
            simulator = Simulator(
                simulation_duration_hours=duration_hours,
                time_step_seconds=config.simulation.TIME_STEP_SECONDS,
                enable_detailed_logging=True
            )

            # Set up progress callback
            def progress_callback(progress, current_time, metrics):
                if int(progress) % 10 == 0:  # Every 10%
                    print(f"Progress: {progress:.1f}%, Time: {current_time/3600:.1f}h, "
                          f"Avg Battery: {metrics.average_battery_percentage:.1f}%")

            # Add nodes with different configurations
            nodes = []
            for i in range(node_count):
                if i == 0:
                    # Static scheduler node
                    node = simulator.add_node(
                        node_id=f"static_node_{i}",
                        node_type=NodeType.ADVANCED_SENSOR,
                        scheduler_type="static",
                        initial_charge_percentage=100.0
                    )
                else:
                    # Adaptive scheduler node
                    node = simulator.add_node(
                        node_id=f"adaptive_node_{i}",
                        node_type=NodeType.ADVANCED_SENSOR,
                        scheduler_type="adaptive",
                        initial_charge_percentage=100.0
                    )
                nodes.append(node)

            # Set logger callback
            simulator.set_data_log_callback(logger.log_entry)
            simulator.set_progress_callback(progress_callback)

            # Run simulation
            start_time = time.time()
            results = simulator.run(real_time_factor=1000.0)  # 1000x speed
            end_time = time.time()

            # Add execution time to results
            results['execution_time_seconds'] = end_time - start_time
            results['logger_statistics'] = logger.get_statistics()

            print(f"Simulation completed in {end_time - start_time:.2f} seconds")

            return results

    def run_comparative_simulation(self, duration_hours: float = 24.0) -> Dict[str, Any]:
        """
        Run comparative simulation between static and adaptive schedulers

        Args:
            duration_hours: Simulation duration in hours

        Returns:
            Comparative results
        """
        print(f"Running comparative simulation: {duration_hours}h")

        results = {
            'simulation_type': 'comparative',
            'duration_hours': duration_hours,
            'static_scheduler': None,
            'adaptive_scheduler': None,
            'comparison': None
        }

        # Run static scheduler simulation
        print("Running static scheduler simulation...")
        static_results = self._run_scheduler_simulation('static', duration_hours)
        results['static_scheduler'] = static_results

        # Run adaptive scheduler simulation
        print("Running adaptive scheduler simulation...")
        adaptive_results = self._run_scheduler_simulation('adaptive', duration_hours)
        results['adaptive_scheduler'] = adaptive_results

        # Generate comparison
        results['comparison'] = self._compare_scheduler_results(
            static_results, adaptive_results
        )

        return results

    def run_scalability_test(self, node_counts: List[int] = [1, 5, 10, 25, 50],
                           duration_hours: float = 12.0) -> Dict[str, Any]:
        """
        Run scalability test with different node counts

        Args:
            node_counts: List of node counts to test
            duration_hours: Simulation duration for each test

        Returns:
            Scalability test results
        """
        print(f"Running scalability test: {node_counts} nodes, {duration_hours}h each")

        results = {
            'simulation_type': 'scalability',
            'test_parameters': {
                'node_counts': node_counts,
                'duration_hours': duration_hours
            },
            'node_results': {},
            'performance_analysis': {}
        }

        for node_count in node_counts:
            print(f"Testing with {node_count} nodes...")
            start_time = time.time()

            # Run simulation with specified node count
            test_results = self.run_basic_simulation(duration_hours, node_count)
            end_time = time.time()

            # Store results
            results['node_results'][node_count] = {
                'simulation_results': test_results,
                'execution_time': end_time - start_time,
                'performance_per_node': (end_time - start_time) / node_count
            }

        # Analyze scalability
        results['performance_analysis'] = self._analyze_scalability(results['node_results'])

        return results

    def run_battery_lifetime_test(self, initial_charges: List[float] = [100, 75, 50, 25],
                                duration_hours: float = 72.0) -> Dict[str, Any]:
        """
        Run battery lifetime test with different initial charge levels

        Args:
            initial_charges: List of initial battery percentages
            duration_hours: Simulation duration

        Returns:
            Battery lifetime test results
        """
        print(f"Running battery lifetime test: {initial_charges}% charges, {duration_hours}h")

        results = {
            'simulation_type': 'battery_lifetime',
            'test_parameters': {
                'initial_charges': initial_charges,
                'duration_hours': duration_hours
            },
            'charge_results': {},
            'lifetime_analysis': {}
        }

        for initial_charge in initial_charges:
            print(f"Testing with {initial_charge}% initial charge...")
            charge_results = self._run_battery_lifetime_simulation(initial_charge, duration_hours)
            results['charge_results'][initial_charge] = charge_results

        # Analyze battery lifetime
        results['lifetime_analysis'] = self._analyze_battery_lifetime(results['charge_results'])

        return results

    def run_environmental_stress_test(self, extreme_conditions: bool = True,
                                   duration_hours: float = 24.0) -> Dict[str, Any]:
        """
        Run environmental stress test

        Args:
            extreme_conditions: Use extreme environmental conditions
            duration_hours: Simulation duration

        Returns:
            Environmental stress test results
        """
        print(f"Running environmental stress test: extreme={extreme_conditions}, {duration_hours}h")

        # Create logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_base_path / "logs" / f"stress_test_{timestamp}.csv"

        with LoggingContext(log_file_path=log_file) as logger:
            # Create simulator with environmental stress
            simulator = Simulator(
                simulation_duration_hours=duration_hours,
                time_step_seconds=config.simulation.TIME_STEP_SECONDS,
                enable_detailed_logging=True
            )

            # Configure extreme conditions if requested
            if extreme_conditions:
                simulator.environment_simulator.base_temperature = 35.0  # High temperature
                simulator.environment_simulator.enable_weather_events = True
                simulator.environment_simulator.trigger_weather_event(0.0)  # Immediate storm

            # Add test nodes
            for i in range(5):
                simulator.add_node(
                    node_id=f"stress_node_{i}",
                    node_type=NodeType.ADVANCED_SENSOR,
                    scheduler_type="adaptive",
                    initial_charge_percentage=75.0
                )

            # Set callbacks
            simulator.set_data_log_callback(logger.log_entry)

            def stress_progress_callback(progress, current_time, metrics):
                if int(progress) % 20 == 0:
                    print(f"Stress test progress: {progress:.1f}%, Nodes depleted: {metrics.nodes_depleted}")

            simulator.set_progress_callback(stress_progress_callback)

            # Run stress test
            results = simulator.run(real_time_factor=500.0)  # Slower for detailed monitoring
            results['test_type'] = 'environmental_stress'
            results['extreme_conditions'] = extreme_conditions
            results['logger_statistics'] = logger.get_statistics()

            return results

    def _run_scheduler_simulation(self, scheduler_type: str, duration_hours: float) -> Dict[str, Any]:
        """Run simulation with specific scheduler type"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_base_path / "logs" / f"{scheduler_type}_scheduler_{timestamp}.csv"

        with LoggingContext(log_file_path=log_file) as logger:
            simulator = Simulator(
                simulation_duration_hours=duration_hours,
                time_step_seconds=config.simulation.TIME_STEP_SECONDS,
                enable_detailed_logging=True
            )

            # Add multiple nodes with same scheduler
            for i in range(3):
                simulator.add_node(
                    node_id=f"{scheduler_type}_node_{i}",
                    node_type=NodeType.ADVANCED_SENSOR,
                    scheduler_type=scheduler_type,
                    initial_charge_percentage=100.0
                )

            simulator.set_data_log_callback(logger.log_entry)
            results = simulator.run(real_time_factor=1000.0)
            results['scheduler_type'] = scheduler_type
            results['logger_statistics'] = logger.get_statistics()

            return results

    def _run_battery_lifetime_simulation(self, initial_charge: float, duration_hours: float) -> Dict[str, Any]:
        """Run battery lifetime simulation with specific initial charge"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_base_path / "logs" / f"battery_{initial_charge}pct_{timestamp}.csv"

        with LoggingContext(log_file_path=log_file) as logger:
            simulator = Simulator(
                simulation_duration_hours=duration_hours,
                time_step_seconds=config.simulation.TIME_STEP_SECONDS,
                enable_detailed_logging=True
            )

            # Add node with specified initial charge
            simulator.add_node(
                node_id=f"battery_test_{initial_charge}pct",
                node_type=NodeType.ADVANCED_SENSOR,
                scheduler_type="adaptive",
                initial_charge_percentage=initial_charge
            )

            simulator.set_data_log_callback(logger.log_entry)
            results = simulator.run(real_time_factor=1000.0)
            results['initial_charge_percentage'] = initial_charge
            results['final_battery_percentage'] = results['overall_metrics']['average_battery_percentage']
            results['battery_depletion_detected'] = results['overall_metrics']['nodes_depleted'] > 0

            return results

    def _compare_scheduler_results(self, static_results: Dict, adaptive_results: Dict) -> Dict[str, Any]:
        """Compare results between static and adaptive schedulers"""
        comparison = {
            'energy_efficiency': {},
            'wake_up_patterns': {},
            'battery_performance': {},
            'overall_improvement': {}
        }

        # Energy efficiency comparison
        static_energy = static_results['overall_metrics']['total_energy_consumed_mah']
        adaptive_energy = adaptive_results['overall_metrics']['total_energy_consumed_mah']

        if static_energy > 0:
            energy_savings = ((static_energy - adaptive_energy) / static_energy) * 100
            comparison['energy_efficiency'] = {
                'static_energy_mah': static_energy,
                'adaptive_energy_mah': adaptive_energy,
                'energy_savings_percentage': energy_savings,
                'more_efficient': 'adaptive' if energy_savings > 0 else 'static'
            }

        # Wake-up pattern comparison
        static_wakeups = static_results['overall_metrics']['total_wakeups']
        adaptive_wakeups = adaptive_results['overall_metrics']['total_wakeups']

        if static_wakeups > 0:
            wakeup_reduction = ((static_wakeups - adaptive_wakeups) / static_wakeups) * 100
            comparison['wake_up_patterns'] = {
                'static_wakeups': static_wakeups,
                'adaptive_wakeups': adaptive_wakeups,
                'wakeup_reduction_percentage': wakeup_reduction,
                'fewer_wakeups': 'adaptive' if wakeup_reduction > 0 else 'static'
            }

        # Battery performance comparison
        static_battery = static_results['overall_metrics']['average_battery_percentage']
        adaptive_battery = adaptive_results['overall_metrics']['average_battery_percentage']

        comparison['battery_performance'] = {
            'static_final_battery': static_battery,
            'adaptive_final_battery': adaptive_battery,
            'battery_advantage': adaptive_battery - static_battery,
            'better_battery_performance': 'adaptive' if adaptive_battery > static_battery else 'static'
        }

        # Overall improvement score
        improvement_score = 0
        factors = []

        if 'energy_savings_percentage' in comparison['energy_efficiency']:
            improvement_score += comparison['energy_efficiency']['energy_savings_percentage']
            factors.append(f"Energy: {comparison['energy_efficiency']['energy_savings_percentage']:.1f}%")

        if 'wakeup_reduction_percentage' in comparison['wake_up_patterns']:
            improvement_score += comparison['wake_up_patterns']['wakeup_reduction_percentage']
            factors.append(f"Wakeups: {comparison['wake_up_patterns']['wakeup_reduction_percentage']:.1f}%")

        comparison['overall_improvement'] = {
            'improvement_score': improvement_score,
            'contributing_factors': factors,
            'recommendation': 'adaptive' if improvement_score > 0 else 'static'
        }

        return comparison

    def _analyze_scalability(self, node_results: Dict) -> Dict[str, Any]:
        """Analyze scalability test results"""
        analysis = {
            'performance_trend': 'linear',
            'efficiency_metrics': {},
            'recommendations': []
        }

        node_counts = sorted(node_results.keys())
        execution_times = [node_results[count]['execution_time'] for count in node_counts]
        per_node_times = [node_results[count]['performance_per_node'] for count in node_counts]

        # Calculate efficiency
        if len(node_counts) >= 2:
            # Simple linear regression to check scalability trend
            n = len(node_counts)
            sum_x = sum(node_counts)
            sum_y = sum(execution_times)
            sum_xy = sum(node_counts[i] * execution_times[i] for i in range(n))
            sum_x2 = sum(x ** 2 for x in node_counts)

            if n * sum_x2 - sum_x ** 2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                analysis['performance_trend'] = 'linear' if abs(slope - per_node_times[0]) < 0.1 else 'non_linear'

        analysis['efficiency_metrics'] = {
            'max_nodes_tested': max(node_counts),
            'max_execution_time': max(execution_times),
            'min_performance_per_node': min(per_node_times),
            'max_performance_per_node': max(per_node_times)
        }

        # Generate recommendations
        if max(per_node_times) / min(per_node_times) > 2.0:
            analysis['recommendations'].append("Performance degrades significantly with more nodes")

        if max(execution_times) > 300:  # 5 minutes
            analysis['recommendations'].append("Consider optimizing for larger node counts")

        return analysis

    def _analyze_battery_lifetime(self, charge_results: Dict) -> Dict[str, Any]:
        """Analyze battery lifetime test results"""
        analysis = {
            'lifetime_projections': {},
            'depletion_patterns': {},
            'efficiency_analysis': {}
        }

        for charge, results in charge_results.items():
            final_battery = results['final_battery_percentage']
            energy_consumed = results['overall_metrics']['total_energy_consumed_mah']
            depletion_detected = results['battery_depletion_detected']

            analysis['lifetime_projections'][charge] = {
                'final_battery_percentage': final_battery,
                'battery_used_percentage': 100 - final_battery,
                'energy_consumed_mah': energy_consumed,
                'depletion_detected': depletion_detected
            }

        return analysis

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Save simulation results to file

        Args:
            results: Results dictionary
            filename: Output filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            simulation_type = results.get('simulation_type', 'simulation')
            filename = f"{simulation_type}_results_{timestamp}.json"

        output_path = self.output_base_path / "logs" / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"Results saved to: {output_path}")
        return str(output_path)


def main():
    """Main entry point for the simulation runner"""
    parser = argparse.ArgumentParser(description='IoT Sleep Scheduling Simulation Runner')
    parser.add_argument('--mode', choices=['basic', 'comparative', 'scalability', 'battery', 'stress'],
                       default='basic', help='Simulation mode to run')
    parser.add_argument('--duration', type=float, default=24.0, help='Simulation duration in hours')
    parser.add_argument('--nodes', type=int, default=2, help='Number of nodes for basic simulation')
    parser.add_argument('--output', type=str, help='Output filename for results')
    parser.add_argument('--config', type=str, help='Configuration file path')

    args = parser.parse_args()

    # Load custom configuration if provided
    if args.config:
        config.load_from_file(args.config)

    # Create simulation runner
    runner = SimulationRunner()

    # Run simulation based on mode
    if args.mode == 'basic':
        results = runner.run_basic_simulation(args.duration, args.nodes)
    elif args.mode == 'comparative':
        results = runner.run_comparative_simulation(args.duration)
    elif args.mode == 'scalability':
        results = runner.run_scalability_test(duration_hours=args.duration)
    elif args.mode == 'battery':
        results = runner.run_battery_lifetime_test(duration_hours=args.duration)
    elif args.mode == 'stress':
        results = runner.run_environmental_stress_test(duration_hours=args.duration)
    else:
        print(f"Unknown mode: {args.mode}")
        return

    # Save results
    result_path = runner.save_results(results, args.output)

    # Print summary
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Duration: {args.duration} hours")
    if args.mode == 'basic':
        print(f"Nodes: {args.nodes}")
    print(f"Results saved to: {result_path}")

    if 'overall_metrics' in results:
        metrics = results['overall_metrics']
        print(f"Total wakeups: {metrics.get('total_wakeups', 0)}")
        print(f"Total transmissions: {metrics.get('total_transmissions_all_nodes', 0)}")
        print(f"Total energy consumed: {metrics.get('total_energy_consumed_all_nodes', 0):.2f} mAh")
        print(f"Average battery: {metrics.get('average_battery_percentage', 0):.1f}%")
        print(f"Nodes depleted: {metrics.get('nodes_depleted', 0)}")

    print("="*50)


if __name__ == "__main__":
    main()