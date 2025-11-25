"""
Simulation Engine for IoT Sleep Scheduling System
Main simulation controller coordinating all components with time management and data collection
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta

from ..models.node import IoTNode, NodeType
from ..config import config


@dataclass
class SimulationEvent:
    """Event in the simulation timeline"""
    timestamp: float
    event_type: str
    node_id: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    """Environmental conditions at a specific time"""
    timestamp: float
    temperature: float
    humidity: float
    light: float
    wind_speed: float = 0.0
    weather_conditions: str = "clear"


@dataclass
class SimulationMetrics:
    """Overall simulation performance metrics"""
    total_simulation_time: float = 0.0
    total_nodes_simulated: int = 0
    total_wakeups_all_nodes: int = 0
    total_transmissions_all_nodes: int = 0
    total_energy_consumed_all_nodes: float = 0.0
    average_battery_percentage: float = 100.0
    nodes_depleted: int = 0
    simulation_steps: int = 0
    events_generated: int = 0
    data_points_collected: int = 0


class EnvironmentSimulator:
    """
    Environmental condition simulator for realistic IoT node testing

    Features:
    - Daily temperature cycles
    - Weather pattern simulation
    - Seasonal variations
    - Random environmental events
    - Location-based parameters
    """

    def __init__(self,
                 base_temperature: float = 20.0,
                 base_humidity: float = 50.0,
                 latitude: float = 40.0,
                 enable_weather_events: bool = None):
        """
        Initialize environment simulator

        Args:
            base_temperature: Base temperature in Celsius
            base_humidity: Base humidity percentage
            latitude: Geographic latitude for sun calculations
            enable_weather_events: Enable random weather events
        """
        self.base_temperature = base_temperature
        self.base_humidity = base_humidity
        self.latitude = latitude
        self.enable_weather_events = enable_weather_events if enable_weather_events is not None else config.simulation.ENABLE_WEATHER_EVENTS

        # Weather patterns
        self.current_weather = "clear"
        self.weather_duration = random.uniform(4, 12) * 3600  # 4-12 hours
        self.next_weather_change = 0.0

        # Seasonal variation
        self.season_offset = 0.0  # Day of year offset for seasonal effects

        # Random event tracking
        self.last_event_time = 0.0

    def get_environment_state(self, current_time: float) -> EnvironmentState:
        """
        Get current environmental conditions

        Args:
            current_time: Current simulation time in seconds

        Returns:
            EnvironmentState with current conditions
        """
        # Calculate time-based factors
        hours = (current_time / 3600) % 24
        day_of_year = (current_time / (24 * 3600)) % 365

        # Temperature calculation with daily and seasonal cycles
        daily_temp_variation = 10 * math.sin(2 * math.pi * (hours - 6) / 24)
        seasonal_temp_variation = 5 * math.sin(2 * math.pi * day_of_year / 365)
        weather_temp_offset = self._get_weather_temperature_offset()

        temperature = (self.base_temperature + daily_temp_variation +
                     seasonal_temp_variation + weather_temp_offset)

        # Humidity calculation (inversely related to temperature)
        humidity_base = self.base_humidity
        humidity_temp_effect = -0.5 * (temperature - self.base_temperature)
        humidity_weather_effect = self._get_weather_humidity_effect()

        humidity = max(20, min(90, humidity_base + humidity_temp_effect + humidity_weather_effect))

        # Light calculation based on sun position
        light_level = self._calculate_light_level(hours, day_of_year)

        # Apply weather effects to light
        light_weather_factor = self._get_weather_light_factor()
        light_level *= light_weather_factor

        # Wind speed (simplified model)
        wind_speed = random.uniform(0, 15) * (1.5 if self.current_weather == "stormy" else 1.0)

        return EnvironmentState(
            timestamp=current_time,
            temperature=temperature,
            humidity=humidity,
            light=light_level,
            wind_speed=wind_speed,
            weather_conditions=self.current_weather
        )

    def _get_weather_temperature_offset(self) -> float:
        """Get temperature offset based on current weather"""
        weather_offsets = {
            "clear": 0.0,
            "cloudy": -2.0,
            "rainy": -5.0,
            "stormy": -8.0,
            "snowy": -10.0
        }
        return weather_offsets.get(self.current_weather, 0.0)

    def _get_weather_humidity_effect(self) -> float:
        """Get humidity effect based on current weather"""
        weather_effects = {
            "clear": 0.0,
            "cloudy": 10.0,
            "rainy": 25.0,
            "stormy": 30.0,
            "snowy": 5.0
        }
        return weather_effects.get(self.current_weather, 0.0)

    def _get_weather_light_factor(self) -> float:
        """Get light reduction factor based on weather"""
        weather_factors = {
            "clear": 1.0,
            "cloudy": 0.7,
            "rainy": 0.4,
            "stormy": 0.2,
            "snowy": 0.8
        }
        return weather_factors.get(self.current_weather, 1.0)

    def _calculate_light_level(self, hours: float, day_of_year: float) -> float:
        """Calculate light level based on sun position"""
        # Simple sun elevation calculation
        if 6 <= hours <= 18:  # Daylight hours
            sun_angle = math.sin(math.pi * (hours - 6) / 12)

            # Seasonal effect on sun intensity
            seasonal_factor = 0.8 + 0.4 * math.sin(2 * math.pi * (day_of_year - 80) / 365)

            # Base light calculation (0-1000 lux)
            base_light = 500 * sun_angle * seasonal_factor

            # Add ambient light
            ambient_light = 50

            return ambient_light + base_light
        else:
            # Night time
            return random.uniform(0, 10)  # Moonlight, starlight

    def update_weather(self, current_time: float):
        """Update weather conditions if necessary"""
        if current_time >= self.next_weather_change:
            # Change weather
            weather_options = ["clear", "cloudy", "rainy", "stormy"]
            if self.base_temperature < 5:  # Cold conditions
                weather_options.append("snowy")

            # Weighted random selection (clear is most common)
            weights = [0.5, 0.25, 0.15, 0.05] if "snowy" not in weather_options else [0.4, 0.2, 0.15, 0.1, 0.15]

            self.current_weather = random.choices(weather_options, weights=weights)[0]
            self.weather_duration = random.uniform(4, 12) * 3600  # 4-12 hours
            self.next_weather_change = current_time + self.weather_duration

    def trigger_weather_event(self, current_time: float) -> bool:
        """Trigger a random weather event"""
        if not self.enable_weather_events:
            return False

        if current_time - self.last_event_time < 3600:  # Minimum 1 hour between events
            return False

        if random.random() < config.simulation.WEATHER_EVENT_PROBABILITY:
            self.current_weather = random.choice(["stormy", "rainy"])
            self.weather_duration = random.uniform(1, 3) * 3600  # 1-3 hours
            self.next_weather_change = current_time + self.weather_duration
            self.last_event_time = current_time
            return True

        return False


class Simulator:
    """
    Main simulation engine for IoT sleep scheduling system

    Features:
    - Multi-node simulation coordination
    - Time management with accelerated simulation
    - Environmental simulation
    - Event-driven architecture
    - Performance metrics collection
    - Data logging integration
    - Configurable simulation parameters
    """

    def __init__(self,
                 simulation_duration_hours: float = None,
                 time_step_seconds: float = None,
                 enable_detailed_logging: bool = None):
        """
        Initialize simulation engine

        Args:
            simulation_duration_hours: Duration of simulation in hours
            time_step_seconds: Time step for simulation updates
            enable_detailed_logging: Enable detailed event logging
        """
        # Simulation parameters
        self.duration_hours = simulation_duration_hours or config.simulation.SIMULATION_DURATION_HOURS
        self.time_step_seconds = time_step_seconds or config.simulation.TIME_STEP_SECONDS
        self.enable_detailed_logging = enable_detailed_logging if enable_detailed_logging is not None else config.simulation.DETAILED_LOGGING

        # Time management
        self.current_time = 0.0
        self.end_time = self.duration_hours * 3600  # Convert to seconds
        self.total_steps = int(self.end_time / self.time_step_seconds)
        self.current_step = 0

        # Nodes
        self.nodes: Dict[str, IoTNode] = {}
        self.node_types: Dict[str, NodeType] = {}

        # Environment
        self.environment_simulator = EnvironmentSimulator()
        self.current_environment: EnvironmentState = None

        # Events and data collection
        self.event_history: List[SimulationEvent] = []
        self.environmental_history: List[EnvironmentState] = []
        self.node_states_history: List[Dict[str, Any]] = []

        # Metrics
        self.metrics = SimulationMetrics()

        # Callbacks for data logging
        self.data_log_callback: Optional[Callable] = None
        self.progress_callback: Optional[Callable] = None

        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.simulation_start_time = None
        self.simulation_end_time = None

    def add_node(self,
                 node_id: str,
                 node_type: NodeType = NodeType.ADVANCED_SENSOR,
                 scheduler_type: str = "adaptive",
                 battery_capacity_mah: float = None,
                 initial_charge_percentage: float = None) -> IoTNode:
        """
        Add an IoT node to the simulation

        Args:
            node_id: Unique identifier for the node
            node_type: Type of IoT node
            scheduler_type: Scheduler type ("static" or "adaptive")
            battery_capacity_mah: Battery capacity in mAh
            initial_charge_percentage: Initial battery charge percentage

        Returns:
            Created IoT node
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")

        node = IoTNode(
            node_id=node_id,
            node_type=node_type,
            scheduler_type=scheduler_type,
            battery_capacity_mah=battery_capacity_mah,
            initial_charge_percentage=initial_charge_percentage
        )

        self.nodes[node_id] = node
        self.node_types[node_id] = node_type

        self.metrics.total_nodes_simulated += 1

        return node

    def remove_node(self, node_id: str):
        """
        Remove a node from the simulation

        Args:
            node_id: Node identifier to remove
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            del self.node_types[node_id]
            self.metrics.total_nodes_simulated -= 1

    def set_data_log_callback(self, callback: Callable):
        """Set callback function for data logging"""
        self.data_log_callback = callback

    def set_progress_callback(self, callback: Callable):
        """Set callback function for progress updates"""
        self.progress_callback = callback

    def run(self, real_time_factor: float = None) -> Dict[str, Any]:
        """
        Run the complete simulation

        Args:
            real_time_factor: Real-time execution factor (1.0 = real time, 1000.0 = 1000x faster)

        Returns:
            Simulation results and metrics
        """
        if not self.nodes:
            raise ValueError("No nodes added to simulation")

        self.is_running = True
        self.simulation_start_time = time.time()

        real_time_factor = real_time_factor or config.simulation.TIME_ACCELERATION_FACTOR

        print(f"Starting simulation: {self.duration_hours} hours, {self.metrics.total_nodes_simulated} nodes")
        print(f"Time step: {self.time_step_seconds}s, Real-time factor: {real_time_factor}x")

        try:
            # Main simulation loop
            while self.current_time < self.end_time and self.is_running:
                if not self.is_paused:
                    step_start_time = time.time()

                    # Execute simulation step
                    self._execute_simulation_step()

                    # Progress callback
                    if self.progress_callback and self.current_step % 100 == 0:
                        progress = (self.current_step / self.total_steps) * 100
                        self.progress_callback(progress, self.current_time, self.metrics)

                    # Real-time timing control
                    if real_time_factor > 0:
                        step_duration = time.time() - step_start_time
                        expected_duration = self.time_step_seconds / real_time_factor
                        if step_duration < expected_duration:
                            time.sleep(expected_duration - step_duration)

                self.current_step += 1

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nSimulation error: {e}")
            raise
        finally:
            self.simulation_end_time = time.time()
            self.is_running = False

        # Generate final results
        return self._generate_simulation_results()

    def _execute_simulation_step(self):
        """Execute a single simulation step"""
        # Update environment
        self.current_environment = self.environment_simulator.get_environment_state(self.current_time)
        self.environmental_history.append(self.current_environment)

        # Trigger weather events occasionally
        if random.random() < 0.01:  # 1% chance per step
            self.environment_simulator.trigger_weather_event(self.current_time)

        # Update weather conditions
        self.environment_simulator.update_weather(self.current_time)

        # Extract environmental data for nodes
        environment_data = {
            'temperature': self.current_environment.temperature,
            'humidity': self.current_environment.humidity,
            'light': self.current_environment.light,
            'wind_speed': self.current_environment.wind_speed
        }

        # Update all nodes
        step_node_states = {}
        for node_id, node in self.nodes.items():
            # Update node
            node_result = node.update(self.current_time, environment_data)

            # Record node state
            step_node_states[node_id] = {
                'state': node.current_state.value,
                'battery_percentage': node.battery.get_remaining_percentage(),
                'energy_consumed_mah': node.battery.total_energy_consumed_mah,
                'sensor_data': node.current_sensor_data.copy(),
                'events': node_result['events']
            }

            # Update metrics
            self.metrics.total_wakeups_all_nodes += node.metrics.total_wakeups
            self.metrics.total_transmissions_all_nodes += node.metrics.total_transmissions
            self.metrics.total_energy_consumed_all_nodes += node.battery.total_energy_consumed_mah

            # Check for battery depletion
            if node.battery.is_depleted():
                self.metrics.nodes_depleted += 1

            # Log data if callback is set
            if self.data_log_callback:
                log_data = {
                    'timestamp': self.current_time,
                    'node_id': node_id,
                    'mode': node.current_state.value,
                    'energy_consumed_mah': node.battery.total_energy_consumed_mah,
                    'battery_percentage': node.battery.get_remaining_percentage(),
                    'temperature': node.current_sensor_data.get('Temperature', 0),
                    'humidity': node.current_sensor_data.get('Humidity', 0),
                    'light_level': node.current_sensor_data.get('Light', 0),
                    'wakeup_reason': self._get_last_wakeup_reason(node_result),
                    'transmission_successful': len(node.transmission_buffer) > 0
                }
                self.data_log_callback(log_data)

        # Store node states
        self.node_states_history.append(step_node_states)
        self.metrics.data_points_collected += len(self.nodes)

        # Generate simulation events
        if self.enable_detailed_logging:
            self._generate_step_events(step_node_states)

        # Update simulation time
        self.current_time += self.time_step_seconds
        self.metrics.simulation_steps += 1

        # Calculate average battery percentage
        if self.nodes:
            total_battery = sum(node.battery.get_remaining_percentage() for node in self.nodes.values())
            self.metrics.average_battery_percentage = total_battery / len(self.nodes)

    def _generate_step_events(self, node_states: Dict[str, Dict]):
        """Generate events for the current step"""
        for node_id, state_data in node_states.items():
            # Check for significant events
            events = state_data['events']

            for event_type, event_data in events.items():
                event = SimulationEvent(
                    timestamp=self.current_time,
                    event_type=event_type,
                    node_id=node_id,
                    data=event_data
                )
                self.event_history.append(event)
                self.metrics.events_generated += 1

    def _get_last_wakeup_reason(self, node_result: Dict[str, Any]) -> str:
        """Extract wakeup reason from node result"""
        events = node_result.get('events', {})

        # Check for sleep events (which indicate previous wakeup)
        if 'sleep' in events:
            return events['sleep'].get('reason', 'unknown')

        # Check for wakeup events
        if 'wakeup' in events:
            return events['wakeup'].get('reason', 'scheduled')

        return 'active'

    def _generate_simulation_results(self) -> Dict[str, Any]:
        """Generate comprehensive simulation results"""
        self.metrics.total_simulation_time = self.simulation_end_time - self.simulation_start_time if self.simulation_end_time else 0

        # Node-specific results
        node_results = {}
        for node_id, node in self.nodes.items():
            node_results[node_id] = {
                'final_battery_percentage': node.battery.get_remaining_percentage(),
                'total_energy_consumed_mah': node.battery.total_energy_consumed_mah,
                'total_wakeups': node.metrics.total_wakeups,
                'total_transmissions': node.metrics.total_transmissions,
                'total_sensor_readings': node.metrics.total_sensor_readings,
                'active_time_seconds': node.metrics.active_time_seconds,
                'sleep_time_seconds': node.metrics.sleep_time_seconds,
                'deep_sleep_time_seconds': node.metrics.deep_sleep_time_seconds,
                'data_packets_generated': node.metrics.data_packets_generated,
                'missed_events': node.metrics.missed_events,
                'scheduler_summary': node.scheduler.get_summary(),
                'battery_depletion_time': node.metrics.battery_depletion_time,
                'final_state': node.current_state.value
            }

        # Performance analysis
        performance_analysis = self._analyze_performance()

        # Energy analysis
        energy_analysis = self._analyze_energy_consumption()

        # Scheduling analysis
        scheduling_analysis = self._analyze_scheduling_effectiveness()

        return {
            'simulation_parameters': {
                'duration_hours': self.duration_hours,
                'time_step_seconds': self.time_step_seconds,
                'total_steps': self.total_steps,
                'completed_steps': self.current_step,
                'total_nodes': self.metrics.total_nodes_simulated,
                'wall_clock_time': self.metrics.total_simulation_time
            },
            'overall_metrics': {
                'total_wakeups': self.metrics.total_wakeups_all_nodes,
                'total_transmissions': self.metrics.total_transmissions_all_nodes,
                'total_energy_consumed_mah': self.metrics.total_energy_consumed_all_nodes,
                'average_battery_percentage': self.metrics.average_battery_percentage,
                'nodes_depleted': self.metrics.nodes_depleted,
                'events_generated': self.metrics.events_generated,
                'data_points_collected': self.metrics.data_points_collected
            },
            'node_results': node_results,
            'performance_analysis': performance_analysis,
            'energy_analysis': energy_analysis,
            'scheduling_analysis': scheduling_analysis,
            'environmental_summary': self._generate_environmental_summary(),
            'raw_data': {
                'environmental_history': self.environmental_history[-100:],  # Last 100 entries
                'node_states_history': self.node_states_history[-100],      # Last 100 entries
                'event_history': self.event_history[-50]                     # Last 50 events
            }
        }

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall simulation performance"""
        if not self.nodes:
            return {}

        # Calculate average metrics
        total_active_time = sum(node.metrics.active_time_seconds for node in self.nodes.values())
        total_sleep_time = sum(node.metrics.sleep_time_seconds for node in self.nodes.values())
        total_deep_sleep_time = sum(node.metrics.deep_sleep_time_seconds for node in self.nodes.values())

        total_operation_time = total_active_time + total_sleep_time + total_deep_sleep_time

        if total_operation_time == 0:
            return {'error': 'No operation time recorded'}

        return {
            'active_time_percentage': (total_active_time / total_operation_time) * 100,
            'sleep_time_percentage': (total_sleep_time / total_operation_time) * 100,
            'deep_sleep_time_percentage': (total_deep_sleep_time / total_operation_time) * 100,
            'average_wakeups_per_node': self.metrics.total_wakeups_all_nodes / len(self.nodes),
            'average_transmissions_per_node': self.metrics.total_transmissions_all_nodes / len(self.nodes),
            'simulation_efficiency': self.current_step / self.total_steps if self.total_steps > 0 else 0,
            'nodes_survived': len(self.nodes) - self.metrics.nodes_depleted,
            'survival_rate': ((len(self.nodes) - self.metrics.nodes_depleted) / len(self.nodes)) * 100 if self.nodes else 0
        }

    def _analyze_energy_consumption(self) -> Dict[str, Any]:
        """Analyze energy consumption patterns"""
        if not self.nodes:
            return {}

        node_energies = [node.battery.total_energy_consumed_mah for node in self.nodes.values()]
        node_batteries = [node.battery.get_remaining_percentage() for node in self.nodes.values()]

        return {
            'total_energy_consumed_mah': self.metrics.total_energy_consumed_all_nodes,
            'average_energy_per_node_mah': sum(node_energies) / len(node_energies) if node_energies else 0,
            'max_energy_consumed_mah': max(node_energies) if node_energies else 0,
            'min_energy_consumed_mah': min(node_energies) if node_energies else 0,
            'energy_consumption_std': self._calculate_std(node_energies),
            'average_remaining_battery': sum(node_batteries) / len(node_batteries) if node_batteries else 0,
            'battery_depletion_rate': self.metrics.total_energy_consumed_all_nodes / self.duration_hours if self.duration_hours > 0 else 0
        }

    def _analyze_scheduling_effectiveness(self) -> Dict[str, Any]:
        """Analyze scheduling algorithm effectiveness"""
        scheduler_analysis = {}

        for node_id, node in self.nodes.items():
            scheduler_type = type(node.scheduler).__name__
            if scheduler_type not in scheduler_analysis:
                scheduler_analysis[scheduler_type] = {
                    'nodes': 0,
                    'total_wakeups': 0,
                    'average_efficiency': 0.0,
                    'energy_savings': []
                }

            analysis = scheduler_analysis[scheduler_type]
            analysis['nodes'] += 1
            analysis['total_wakeups'] += node.metrics.total_wakeups
            analysis['average_efficiency'] += node.scheduler.get_efficiency_score()

            # Calculate energy savings compared to static baseline
            if hasattr(node.scheduler, 'get_efficiency_score'):
                analysis['energy_savings'].append(node.scheduler.get_efficiency_score())

        # Calculate averages
        for scheduler_type, analysis in scheduler_analysis.items():
            if analysis['nodes'] > 0:
                analysis['average_efficiency'] /= analysis['nodes']
                analysis['average_wakeups_per_node'] = analysis['total_wakeups'] / analysis['nodes']

        return scheduler_analysis

    def _generate_environmental_summary(self) -> Dict[str, Any]:
        """Generate summary of environmental conditions"""
        if not self.environmental_history:
            return {}

        temperatures = [env.temperature for env in self.environmental_history]
        humidities = [env.humidity for env in self.environmental_history]
        light_levels = [env.light for env in self.environmental_history]

        weather_counts = {}
        for env in self.environmental_history:
            weather_counts[env.weather_conditions] = weather_counts.get(env.weather_conditions, 0) + 1

        return {
            'average_temperature': sum(temperatures) / len(temperatures) if temperatures else 0,
            'min_temperature': min(temperatures) if temperatures else 0,
            'max_temperature': max(temperatures) if temperatures else 0,
            'average_humidity': sum(humidities) / len(humidities) if humidities else 0,
            'min_humidity': min(humidities) if humidities else 0,
            'max_humidity': max(humidities) if humidities else 0,
            'average_light_level': sum(light_levels) / len(light_levels) if light_levels else 0,
            'weather_distribution': weather_counts,
            'total_environmental_updates': len(self.environmental_history)
        }

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values or len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def pause(self):
        """Pause simulation"""
        self.is_paused = True

    def resume(self):
        """Resume simulation"""
        self.is_paused = False

    def stop(self):
        """Stop simulation"""
        self.is_running = False

    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            'current_time': self.current_time,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percentage': (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'nodes_count': len(self.nodes),
            'current_environment': self.current_environment.__dict__ if self.current_environment else None
        }