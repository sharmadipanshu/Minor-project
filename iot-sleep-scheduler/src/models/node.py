"""
IoT Sensor Node Model for Sleep Scheduling System
Comprehensive node simulation integrating all components with realistic behavior
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from .battery import Battery
from .sensor import SensorManager, TemperatureSensor, HumiditySensor, LightSensor
from .scheduler import Scheduler, StaticDutyCycleScheduler, AdaptiveDutyCycleScheduler, WakeupReason
from .power_model import PowerModel, PowerState
from ..config import config


class NodeState(Enum):
    """Operational states of IoT node"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    TRANSMITTING = "transmitting"
    SENSING = "sensing"
    PROCESSING = "processing"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class NodeType(Enum):
    """Types of IoT nodes with different characteristics"""
    BASIC_SENSOR = "basic_sensor"
    ADVANCED_SENSOR = "advanced_sensor"
    GATEWAY = "gateway"
    ACTUATOR = "actuator"


@dataclass
class NodeMetrics:
    """Performance metrics for IoT node"""
    total_wakeups: int = 0
    total_transmissions: int = 0
    total_sensor_readings: int = 0
    total_processing_time: float = 0.0
    active_time_seconds: float = 0.0
    sleep_time_seconds: float = 0.0
    deep_sleep_time_seconds: float = 0.0
    transmission_time_seconds: float = 0.0
    energy_consumed_mah: float = 0.0
    average_wakeup_interval: float = 0.0
    data_packets_generated: int = 0
    missed_events: int = 0
    battery_depletion_time: float = 0.0


@dataclass
class TransmissionPacket:
    """Data packet for transmission"""
    timestamp: float
    node_id: str
    sensor_data: Dict[str, float]
    battery_percentage: float
    node_state: str
    packet_size_bytes: int
    transmission_successful: bool = False


class IoTNode:
    """
    Comprehensive IoT sensor node simulation

    Features:
    - Multi-state operation with power management
    - Integrated sensors with realistic data generation
    - Advanced sleep scheduling algorithms
    - Battery modeling with discharge characteristics
    - Power consumption tracking and optimization
    - Data transmission and processing simulation
    - Environmental adaptation
    - Performance metrics collection
    """

    def __init__(self,
                 node_id: str,
                 node_type: NodeType = NodeType.ADVANCED_SENSOR,
                 scheduler_type: str = "adaptive",
                 battery_capacity_mah: float = None,
                 initial_charge_percentage: float = None):
        """
        Initialize IoT sensor node

        Args:
            node_id: Unique identifier for the node
            node_type: Type of IoT node
            scheduler_type: Type of scheduler ("static" or "adaptive")
            battery_capacity_mah: Battery capacity in mAh
            initial_charge_percentage: Initial battery charge percentage
        """
        # Node identification
        self.node_id = node_id
        self.node_type = node_type

        # Initialize components
        self.battery = Battery(
            capacity_mah=battery_capacity_mah,
            initial_charge_percentage=initial_charge_percentage
        )
        self.power_model = PowerModel()
        self.sensor_manager = SensorManager()
        self.scheduler = self._create_scheduler(scheduler_type)

        # Node state
        self.current_state = NodeState.INITIALIZING
        self.last_wakeup_time = 0.0
        self.last_transmission_time = 0.0
        self.last_state_change_time = 0.0
        self.sleep_duration_remaining = 0.0

        # Data storage
        self.current_sensor_data: Dict[str, float] = {}
        self.transmission_buffer: List[TransmissionPacket] = []
        self.recent_sensor_readings: List[Dict[str, float]] = []

        # Metrics
        self.metrics = NodeMetrics()
        self.state_durations: Dict[NodeState, float] = {state: 0.0 for state in NodeState}
        self.state_transition_count: Dict[Tuple[NodeState, NodeState], int] = {}

        # Configuration based on node type
        self._configure_node_type()

        # Initialize sensors
        self._initialize_sensors()

        # Set initial state
        self._transition_to_state(NodeState.ACTIVE, 0.0, "initialization_complete")

    def update(self, current_time: float, environment: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Update node state for current time step

        Args:
            current_time: Current simulation time in seconds
            environment: Environmental conditions

        Returns:
            Dictionary containing update results and events
        """
        # Update battery with current conditions
        ambient_temp = environment.get('temperature', 25.0) if environment else 25.0
        self.battery.update(current_time, ambient_temp)

        # Check battery status
        if self.battery.is_depleted():
            self._transition_to_state(NodeState.SHUTDOWN, current_time, "battery_depleted")
            return self._create_update_result(current_time, {"shutdown": True})

        # State machine
        result = self._process_state_machine(current_time, environment)

        # Update metrics
        self._update_metrics(current_time)

        # Return update results
        return result

    def wakeup(self, current_time: float, reason: WakeupReason):
        """
        Wake up node from sleep state

        Args:
            current_time: Current simulation time
            reason: Reason for wake-up
        """
        if self.current_state in [NodeState.LIGHT_SLEEP, NodeState.DEEP_SLEEP]:
            # Calculate sleep duration
            sleep_duration = current_time - self.last_wakeup_time
            self.metrics.average_wakeup_interval = (
                (self.metrics.average_wakeup_interval * self.metrics.total_wakeups + sleep_duration) /
                (self.metrics.total_wakeups + 1)
            )

            self.last_wakeup_time = current_time
            self.metrics.total_wakeups += 1

            # Record wake-up with scheduler
            self.scheduler.record_wakeup(current_time, reason)

            # Transition to active state
            self._transition_to_state(NodeState.ACTIVE, current_time, f"wakeup_{reason.value}")

    def sleep(self, duration_seconds: float):
        """
        Put node to sleep

        Args:
            duration_seconds: Sleep duration in seconds
        """
        if self.current_state == NodeState.ACTIVE:
            # Determine sleep mode based on battery level and duration
            if self.battery.is_low() or duration_seconds > 600:  # 10 minutes
                sleep_state = NodeState.DEEP_SLEEP
            else:
                sleep_state = NodeState.LIGHT_SLEEP

            self.sleep_duration_remaining = duration_seconds
            self._transition_to_state(sleep_state, self.last_wakeup_time, f"sleep_{duration_seconds}s")

    def sample_sensors(self, current_time: float, environment: Dict[str, float] = None) -> Dict[str, float]:
        """
        Sample all sensors and return readings

        Args:
            current_time: Current simulation time
            environment: Environmental conditions

        Returns:
            Dictionary of sensor readings
        """
        if self.current_state not in [NodeState.ACTIVE, NodeState.SENSING]:
            return {}

        # Transition to sensing state temporarily
        original_state = self.current_state
        self._transition_to_state(NodeState.SENSING, current_time, "sensor_sampling")

        # Read all sensors
        readings = self.sensor_manager.read_all_sensors(current_time, environment)
        self.current_sensor_data = {name: reading.value for name, reading in readings.items()}

        # Store recent readings for trend analysis
        self.recent_sensor_readings.append(self.current_sensor_data.copy())
        if len(self.recent_sensor_readings) > 100:  # Keep last 100 readings
            self.recent_sensor_readings = self.recent_sensor_readings[-100:]

        # Update metrics
        self.metrics.total_sensor_readings += len(readings)

        # Calculate energy consumption for sensing
        num_sensors = len([r for r in readings.values() if r.is_valid])
        sensing_energy = self.power_model.calculate_sensing_energy(num_sensors, self.battery.temperature_celsius)
        self.battery.consume_current(self.power_model.get_current_draw(PowerState.SENSING),
                                   self.power_model.tx_duration)

        # Return to original state
        self._transition_to_state(original_state, current_time, "sensor_sampling_complete")

        return self.current_sensor_data

    def transmit_data(self, current_time: float, data: Dict[str, float] = None) -> bool:
        """
        Transmit sensor data

        Args:
            current_time: Current simulation time
            data: Data to transmit (uses current sensor data if not provided)

        Returns:
            True if transmission successful, False otherwise
        """
        if self.current_state != NodeState.ACTIVE:
            return False

        # Use current sensor data if not provided
        if data is None:
            data = self.current_sensor_data.copy()

        # Create transmission packet
        packet_size = self._calculate_packet_size(data)
        packet = TransmissionPacket(
            timestamp=current_time,
            node_id=self.node_id,
            sensor_data=data,
            battery_percentage=self.battery.get_remaining_percentage(),
            node_state=self.current_state.value,
            packet_size_bytes=packet_size
        )

        # Transition to transmitting state
        self._transition_to_state(NodeState.TRANSMITTING, current_time, "data_transmission")

        # Calculate transmission energy
        transmission_energy = self.power_model.calculate_transmission_energy(
            packet_size, self.battery.temperature_celsius
        )

        # Simulate transmission
        transmission_success = self._simulate_transmission()

        # Update packet
        packet.transmission_successful = transmission_success

        if transmission_success:
            self.transmission_buffer.append(packet)
            self.last_transmission_time = current_time
            self.metrics.total_transmissions += 1
            self.metrics.data_packets_generated += 1
        else:
            self.metrics.missed_events += 1

        # Return to active state
        self._transition_to_state(NodeState.ACTIVE, current_time, "transmission_complete")

        return transmission_success

    def process_data(self, current_time: float) -> Dict[str, Any]:
        """
        Process collected sensor data

        Args:
            current_time: Current simulation time

        Returns:
            Processing results
        """
        if self.current_state != NodeState.ACTIVE or not self.current_sensor_data:
            return {}

        # Transition to processing state
        self._transition_to_state(NodeState.PROCESSING, current_time, "data_processing")

        # Simulate processing time
        processing_time = random.uniform(0.05, 0.2)  # 50-200ms

        # Calculate processing energy
        processing_energy = self.power_model.calculate_processing_energy(
            processing_time, self.battery.temperature_celsius
        )

        # Consume energy for processing
        self.battery.consume_current(self.power_model.get_current_draw(PowerState.PROCESSING),
                                   processing_time)

        # Update metrics
        self.metrics.total_processing_time += processing_time

        # Simple data analysis
        results = {
            'processing_time': processing_time,
            'energy_consumed_mah': processing_energy,
            'data_analyzed': len(self.current_sensor_data),
            'battery_at_processing': self.battery.get_remaining_percentage(),
            'analysis_results': self._analyze_sensor_data()
        }

        # Return to active state
        self._transition_to_state(NodeState.ACTIVE, current_time, "processing_complete")

        return results

    def get_node_state(self) -> Dict[str, Any]:
        """
        Get comprehensive node state information

        Returns:
            Dictionary containing node state
        """
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'current_state': self.current_state.value,
            'battery_percentage': self.battery.get_remaining_percentage(),
            'battery_voltage': self.battery.get_voltage(),
            'battery_remaining_mah': self.battery.get_remaining_capacity_mah(),
            'energy_consumed_mah': self.battery.total_energy_consumed_mah,
            'last_wakeup_time': self.last_wakeup_time,
            'last_transmission_time': self.last_transmission_time,
            'current_sensor_data': self.current_sensor_data,
            'scheduler_metrics': self.scheduler.get_summary(),
            'metrics': {
                'total_wakeups': self.metrics.total_wakeups,
                'total_transmissions': self.metrics.total_transmissions,
                'total_sensor_readings': self.metrics.total_sensor_readings,
                'active_time_seconds': self.metrics.active_time_seconds,
                'sleep_time_seconds': self.metrics.sleep_time_seconds,
                'data_packets_generated': self.metrics.data_packets_generated
            }
        }

    def get_scheduler_decision(self, current_time: float) -> Dict[str, Any]:
        """
        Get scheduler decision for current state

        Args:
            current_time: Current simulation time

        Returns:
            Scheduler decision information
        """
        node_state = {
            'battery_percentage': self.battery.get_remaining_percentage(),
            'sensor_data': self.current_sensor_data,
            'sensor_changes': self.sensor_manager.detect_significant_changes(),
            'activity_level': self.sensor_manager.get_overall_activity_level(),
            'current_state': self.current_state.value,
            'time_since_wakeup': current_time - self.last_wakeup_time
        }

        decision = self.scheduler.should_wakeup(current_time, node_state)
        sleep_duration = self.scheduler.get_sleep_duration(current_time, node_state)

        return {
            'should_wakeup': decision.should_wakeup,
            'sleep_duration_minutes': sleep_duration,
            'wakeup_reason': decision.wakeup_reason.value if decision.should_wakeup else None,
            'confidence': decision.confidence,
            'metadata': decision.metadata
        }

    def reset(self):
        """Reset node to initial state"""
        self.battery.reset()
        self.sensor_manager.reset_all_sensors()
        self.scheduler.reset_metrics()
        self.current_sensor_data.clear()
        self.transmission_buffer.clear()
        self.recent_sensor_readings.clear()
        self.metrics = NodeMetrics()
        self.state_durations = {state: 0.0 for state in NodeState}
        self.state_transition_count.clear()
        self.last_wakeup_time = 0.0
        self.last_transmission_time = 0.0
        self.sleep_duration_remaining = 0.0
        self._transition_to_state(NodeState.ACTIVE, 0.0, "reset_complete")

    def _create_scheduler(self, scheduler_type: str) -> Scheduler:
        """Create scheduler based on type"""
        if scheduler_type.lower() == "static":
            return StaticDutyCycleScheduler()
        else:
            return AdaptiveDutyCycleScheduler()

    def _configure_node_type(self):
        """Configure node parameters based on node type"""
        type_configs = {
            NodeType.BASIC_SENSOR: {
                'sensor_count': 1,
                'processing_capability': 0.5,
                'transmission_power': 0.7,
                'battery_drain_factor': 1.0
            },
            NodeType.ADVANCED_SENSOR: {
                'sensor_count': 3,
                'processing_capability': 1.0,
                'transmission_power': 1.0,
                'battery_drain_factor': 1.0
            },
            NodeType.GATEWAY: {
                'sensor_count': 2,
                'processing_capability': 1.5,
                'transmission_power': 1.5,
                'battery_drain_factor': 1.5
            },
            NodeType.ACTUATOR: {
                'sensor_count': 1,
                'processing_capability': 0.8,
                'transmission_power': 0.8,
                'battery_drain_factor': 1.2
            }
        }

        self.config = type_configs.get(self.node_type, type_configs[NodeType.ADVANCED_SENSOR])

    def _initialize_sensors(self):
        """Initialize sensors based on node configuration"""
        sensor_count = self.config['sensor_count']

        if sensor_count >= 1:
            self.sensor_manager.add_sensor(TemperatureSensor())
        if sensor_count >= 2:
            self.sensor_manager.add_sensor(HumiditySensor())
        if sensor_count >= 3:
            self.sensor_manager.add_sensor(LightSensor())

    def _process_state_machine(self, current_time: float, environment: Dict[str, float] = None) -> Dict[str, Any]:
        """Process node state machine"""
        events = {}

        # Update state durations
        if self.last_state_change_time < current_time:
            duration = current_time - self.last_state_change_time
            self.state_durations[self.current_state] += duration

        # State-specific processing
        if self.current_state == NodeState.ACTIVE:
            events = self._process_active_state(current_time, environment)
        elif self.current_state == NodeState.LIGHT_SLEEP:
            events = self._process_light_sleep_state(current_time)
        elif self.current_state == NodeState.DEEP_SLEEP:
            events = self._process_deep_sleep_state(current_time)
        elif self.current_state == NodeState.INITIALIZING:
            self._transition_to_state(NodeState.ACTIVE, current_time, "initialization_complete")

        return self._create_update_result(current_time, events)

    def _process_active_state(self, current_time: float, environment: Dict[str, float] = None) -> Dict[str, Any]:
        """Process active state operations"""
        events = {}

        # Sample sensors
        if not self.current_sensor_data:
            sensor_readings = self.sample_sensors(current_time, environment)
            events['sensor_readings'] = sensor_readings

        # Process data
        if self.current_sensor_data:
            processing_results = self.process_data(current_time)
            if processing_results:
                events['data_processing'] = processing_results

        # Transmit data periodically
        time_since_tx = current_time - self.last_transmission_time
        if time_since_tx > config.get_seconds_from_minutes(10):  # Transmit every 10 minutes
            transmission_success = self.transmit_data(current_time)
            events['transmission'] = {
                'success': transmission_success,
                'time_since_last_tx': time_since_tx
            }

        # Check scheduler for sleep decision
        scheduler_decision = self.get_scheduler_decision(current_time)
        if scheduler_decision['should_wakeup']:
            self.wakeup(current_time, WakeupReason(scheduler_decision['wakeup_reason']))
        else:
            # Go to sleep
            sleep_duration = config.get_seconds_from_minutes(scheduler_decision['sleep_duration_minutes'])
            self.sleep(sleep_duration)
            events['sleep'] = {
                'duration_seconds': sleep_duration,
                'reason': scheduler_decision.get('wakeup_reason', 'scheduled')
            }

        return events

    def _process_light_sleep_state(self, current_time: float) -> Dict[str, Any]:
        """Process light sleep state"""
        events = {}

        # Consume sleep power
        sleep_current = self.power_model.get_current_draw(PowerState.LIGHT_SLEEP)
        time_delta = current_time - self.last_state_change_time
        energy_consumed = self.battery.consume_current(sleep_current, time_delta)

        # Check if sleep duration is complete
        if self.sleep_duration_remaining > 0:
            self.sleep_duration_remaining -= time_delta

        if self.sleep_duration_remaining <= 0:
            # Check scheduler for wake-up decision
            scheduler_decision = self.get_scheduler_decision(current_time)
            if scheduler_decision['should_wakeup']:
                reason = WakeupReason(scheduler_decision['wakeup_reason']) if scheduler_decision['wakeup_reason'] else WakeupReason.SCHEDULED_TIMER
                self.wakeup(current_time, reason)
                events['wakeup'] = {'reason': reason.value}

        return events

    def _process_deep_sleep_state(self, current_time: float) -> Dict[str, Any]:
        """Process deep sleep state"""
        events = {}

        # Consume deep sleep power
        sleep_current = self.power_model.get_current_draw(PowerState.DEEP_SLEEP)
        time_delta = current_time - self.last_state_change_time
        energy_consumed = self.battery.consume_current(sleep_current, time_delta)

        # Check if sleep duration is complete
        if self.sleep_duration_remaining > 0:
            self.sleep_duration_remaining -= time_delta

        if self.sleep_duration_remaining <= 0:
            # Only wake up on timer in deep sleep
            scheduler_decision = self.get_scheduler_decision(current_time)
            if scheduler_decision['should_wakeup'] and scheduler_decision['wakeup_reason'] == 'scheduled_timer':
                self.wakeup(current_time, WakeupReason.SCHEDULED_TIMER)
                events['wakeup'] = {'reason': 'scheduled_timer'}

        return events

    def _transition_to_state(self, new_state: NodeState, current_time: float, reason: str):
        """Transition to new state"""
        old_state = self.current_state
        self.current_state = new_state
        self.last_state_change_time = current_time

        # Track state transitions
        transition = (old_state, new_state)
        self.state_transition_count[transition] = self.state_transition_count.get(transition, 0) + 1

        # Update metrics
        if new_state == NodeState.ACTIVE:
            self.metrics.active_time_seconds += (current_time - self.last_state_change_time)
        elif new_state == NodeState.LIGHT_SLEEP:
            self.metrics.sleep_time_seconds += (current_time - self.last_state_change_time)
        elif new_state == NodeState.DEEP_SLEEP:
            self.metrics.deep_sleep_time_seconds += (current_time - self.last_state_change_time)

    def _calculate_packet_size(self, data: Dict[str, float]) -> int:
        """Calculate packet size in bytes"""
        # Base packet overhead
        base_size = 20  # Headers, metadata, etc.

        # Data size (4 bytes per float + key size)
        data_size = sum(4 + len(key.encode('utf-8')) for key in data.keys())

        # Additional metadata
        metadata_size = 16  # Timestamp, node ID, etc.

        return base_size + data_size + metadata_size

    def _simulate_transmission(self) -> bool:
        """Simulate data transmission with success probability"""
        # Success probability depends on battery level
        battery_percentage = self.battery.get_remaining_percentage()

        if battery_percentage > 50:
            success_probability = 0.98
        elif battery_percentage > 20:
            success_probability = 0.90
        else:
            success_probability = 0.75

        return random.random() < success_probability

    def _analyze_sensor_data(self) -> Dict[str, Any]:
        """Simple analysis of sensor data"""
        if not self.current_sensor_data:
            return {}

        analysis = {}

        # Temperature analysis
        if 'Temperature' in self.current_sensor_data:
            temp = self.current_sensor_data['Temperature']
            analysis['temperature_status'] = (
                'high' if temp > config.scheduling.THRESHOLD_TEMP_HIGH else
                'low' if temp < config.scheduling.THRESHOLD_TEMP_LOW else
                'normal'
            )

        # Humidity analysis
        if 'Humidity' in self.current_sensor_data:
            humidity = self.current_sensor_data['Humidity']
            analysis['humidity_status'] = (
                'high' if humidity > config.scheduling.THRESHOLD_HUMIDITY_HIGH else
                'low' if humidity < config.scheduling.THRESHOLD_HUMIDITY_LOW else
                'normal'
            )

        # Light analysis
        if 'Light' in self.current_sensor_data:
            light = self.current_sensor_data['Light']
            analysis['light_status'] = (
                'bright' if light > config.scheduling.THRESHOLD_LIGHT_HIGH else
                'dark' if light < config.scheduling.THRESHOLD_LIGHT_LOW else
                'normal'
            )

        return analysis

    def _update_metrics(self, current_time: float):
        """Update node metrics"""
        # Update energy consumption
        self.metrics.energy_consumed_mah = self.battery.total_energy_consumed_mah

        # Track battery depletion time
        if self.battery.is_depleted() and self.metrics.battery_depletion_time == 0:
            self.metrics.battery_depletion_time = current_time

    def _create_update_result(self, current_time: float, events: Dict[str, Any]) -> Dict[str, Any]:
        """Create update result dictionary"""
        return {
            'timestamp': current_time,
            'node_id': self.node_id,
            'current_state': self.current_state.value,
            'battery_percentage': self.battery.get_remaining_percentage(),
            'battery_voltage': self.battery.get_voltage(),
            'energy_consumed_mah': self.battery.total_energy_consumed_mah,
            'events': events,
            'sensor_data': self.current_sensor_data.copy(),
            'metrics': {
                'total_wakeups': self.metrics.total_wakeups,
                'total_transmissions': self.metrics.total_transmissions,
                'active_time_seconds': self.metrics.active_time_seconds
            }
        }