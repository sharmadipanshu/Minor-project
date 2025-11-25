"""
Configuration parameters for IoT Sleep Scheduling and Power Optimization
Centralized configuration management for all simulation components
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class PowerConfig:
    """Power consumption parameters for different operational modes"""
    # Current consumption in different modes (configurable ranges as specified)
    ACTIVE_CURRENT_MA: float = 15.0  # Range: 10-25 mA
    SLEEP_CURRENT_MA: float = 0.2    # 200µA, Range: 50-300µA (0.05-0.3 mA)
    DEEP_SLEEP_CURRENT_MA: float = 0.01  # 10µA, Range: 5-20µA (0.005-0.02 mA)
    TX_CURRENT_MA: float = 50.0      # Range: 30-80 mA

    # Transmission parameters
    TX_DURATION_SECONDS: float = 0.1  # 100ms transmission burst
    TX_PRE_WAKE_ENERGY_MAH: float = 0.001  # Pre-transmission wake-up energy

    # Power calculation constants
    VOLTAGE: float = 3.7  # Typical Li-ion voltage


@dataclass
class BatteryConfig:
    """Battery and energy storage parameters"""
    BATTERY_CAPACITY_MAH: float = 2000.0  # Configurable capacity
    INITIAL_CHARGE_PERCENTAGE: float = 100.0
    CRITICAL_BATTERY_PERCENTAGE: float = 5.0  # Shutdown threshold
    LOW_BATTERY_THRESHOLD: float = 20.0  # Conservation mode threshold

    # Battery characteristics
    SELF_DISCHARGE_RATE_PER_MONTH: float = 0.5  # Percentage per month
    EFFICIENCY_FACTOR: float = 0.95  # Charge/discharge efficiency
    TEMPERATURE_COEFFICIENT: float = 0.002  # Capacity change per degree Celsius

    # Voltage discharge curve (percentage to voltage mapping)
    DISCHARGE_CURVE: Dict[float, float] = None

    def __post_init__(self):
        if self.DISCHARGE_CURVE is None:
            self.DISCHARGE_CURVE = {
                100.0: 4.2,   # Fully charged Li-ion
                90.0: 4.0,
                80.0: 3.9,
                70.0: 3.8,
                60.0: 3.7,
                50.0: 3.6,
                40.0: 3.5,
                30.0: 3.4,
                20.0: 3.3,
                10.0: 3.1,
                5.0: 3.0,    # Critical voltage
                0.0: 2.7      # Fully discharged
            }


@dataclass
class SensorConfig:
    """Sensor simulation parameters and realistic value ranges"""
    # Temperature sensor parameters
    TEMP_MIN: float = -10.0    # Degrees Celsius
    TEMP_MAX: float = 50.0
    TEMP_ACCURACY: float = 0.5  # ±0.5°C accuracy
    TEMP_NOISE_STD: float = 0.2  # Standard deviation for noise
    TEMP_CHANGE_THRESHOLD: float = 2.0  # Degrees change for significant event

    # Humidity sensor parameters
    HUMIDITY_MIN: float = 20.0   # Percentage
    HUMIDITY_MAX: float = 90.0
    HUMIDITY_ACCURACY: float = 3.0  # ±3% accuracy
    HUMIDITY_NOISE_STD: float = 2.0
    HUMIDITY_CHANGE_THRESHOLD: float = 5.0  # Percentage change for significant event

    # Light sensor parameters
    LIGHT_MIN: float = 0.0       # Lux
    LIGHT_MAX: float = 1000.0
    LIGHT_ACCURACY: float = 10.0  # ±10 lux accuracy
    LIGHT_NOISE_STD: float = 5.0
    LIGHT_CHANGE_THRESHOLD: float = 50.0  # Lux change for significant event

    # Common sensor parameters
    SAMPLING_INTERVAL_SECONDS: float = 1.0
    FAILURE_PROBABILITY: float = 0.001  # 0.1% chance of failed reading
    SENSOR_POWER_CONSUMPTION_MAH: float = 0.0001  # Per reading


@dataclass
class SchedulingConfig:
    """Sleep scheduling and duty cycling parameters"""
    # Static duty cycle parameters
    STATIC_DUTY_CYCLE_MINUTES: float = 5.0

    # Adaptive duty cycle parameters
    ADAPTIVE_MIN_SLEEP_MINUTES: float = 1.0
    ADAPTIVE_MAX_SLEEP_MINUTES: float = 30.0
    ADAPTIVE_DEFAULT_SLEEP_MINUTES: float = 5.0

    # Threshold-based wakeup parameters
    BATTERY_LOW_THRESHOLD: float = 20.0  # Percentage
    BATTERY_CRITICAL_THRESHOLD: float = 5.0  # Percentage

    # Environmental thresholds
    THRESHOLD_TEMP_HIGH: float = 30.0  # Degrees Celsius
    THRESHOLD_TEMP_LOW: float = 5.0
    THRESHOLD_HUMIDITY_HIGH: float = 80.0  # Percentage
    THRESHOLD_HUMIDITY_LOW: float = 30.0
    THRESHOLD_LIGHT_HIGH: float = 800.0  # Lux
    THRESHOLD_LIGHT_LOW: float = 10.0

    # Adaptive scheduling parameters
    HIGH_CHANGE_RATE_THRESHOLD: float = 0.5  # Rate of change per minute
    TREND_ANALYSIS_WINDOW: int = 5  # Number of readings for trend analysis
    TREND_THRESHOLD: float = 0.1  # Slope threshold for trend detection

    # Battery-aware scheduling
    BATTERY_CONSERVATION_MULTIPLIER: float = 2.0  # Sleep duration multiplier
    BATTERY_CRITICAL_MULTIPLIER: float = 4.0  # Extreme conservation

    # Time-based adjustments
    NIGHT_SLEEP_EXTENSION: float = 1.5  # Extend sleep during night hours
    DAY_ACTIVITY_BOOST: float = 0.7    # Reduce sleep during active hours


@dataclass
class SimulationConfig:
    """Main simulation engine parameters"""
    # Time parameters
    SIMULATION_DURATION_HOURS: float = 24.0
    TIME_STEP_SECONDS: float = 1.0
    LOG_INTERVAL_SECONDS: float = 60.0

    # Simulation speed
    TIME_ACCELERATION_FACTOR: float = 1000.0  # 1000x real-time

    # Node parameters
    DEFAULT_NUM_NODES: int = 2
    NODE_IDS: list = None

    # Environment simulation
    ENABLE_ENVIRONMENTAL_CHANGES: bool = True
    ENABLE_WEATHER_EVENTS: bool = True
    WEATHER_EVENT_PROBABILITY: float = 0.05  # 5% chance per hour

    # Data logging
    DETAILED_LOGGING: bool = True
    SAVE_SENSOR_HISTORY: bool = True
    SAVE_ENERGY_HISTORY: bool = True

    # Output files
    LOG_FILENAME: str = "simulation_log.csv"
    GRAPHS_OUTPUT_DIR: str = "results/graphs"
    LOGS_OUTPUT_DIR: str = "results/logs"
    SUMMARY_FILENAME: str = "summary_metrics.json"

    def __post_init__(self):
        if self.NODE_IDS is None:
            self.NODE_IDS = ["static_node", "adaptive_node"]


@dataclass
class Config:
    """Main configuration class containing all sub-configurations"""
    power: PowerConfig = PowerConfig()
    battery: BatteryConfig = BatteryConfig()
    sensor: SensorConfig = SensorConfig()
    scheduling: SchedulingConfig = SchedulingConfig()
    simulation: SimulationConfig = SimulationConfig()

    # Derived parameters
    SECONDS_PER_HOUR: int = 3600
    MINUTES_PER_HOUR: int = 60
    HOURS_PER_DAY: int = 24

    def get_seconds_from_minutes(self, minutes: float) -> float:
        """Convert minutes to seconds"""
        return minutes * 60.0

    def get_hours_from_seconds(self, seconds: float) -> float:
        """Convert seconds to hours"""
        return seconds / 3600.0

    def get_total_simulation_steps(self) -> int:
        """Get total number of simulation steps"""
        return int(self.simulation.SIMULATION_DURATION_HOURS * self.SECONDS_PER_HOUR / self.simulation.TIME_STEP_SECONDS)

    def validate(self) -> bool:
        """Validate all configuration parameters"""
        try:
            # Validate power parameters
            assert 10 <= self.power.ACTIVE_CURRENT_MA <= 25, "Active current out of range"
            assert 0.05 <= self.power.SLEEP_CURRENT_MA <= 0.3, "Sleep current out of range"
            assert 0.005 <= self.power.DEEP_SLEEP_CURRENT_MA <= 0.02, "Deep sleep current out of range"
            assert 30 <= self.power.TX_CURRENT_MA <= 80, "TX current out of range"

            # Validate battery parameters
            assert 500 <= self.battery.BATTERY_CAPACITY_MAH <= 10000, "Battery capacity out of reasonable range"
            assert 0 < self.battery.INITIAL_CHARGE_PERCENTAGE <= 100, "Initial charge invalid"

            # Validate sensor ranges
            assert self.sensor.TEMP_MIN < self.sensor.TEMP_MAX, "Temperature range invalid"
            assert 0 <= self.sensor.HUMIDITY_MIN <= self.sensor.HUMIDITY_MAX <= 100, "Humidity range invalid"
            assert 0 <= self.sensor.LIGHT_MIN <= self.sensor.LIGHT_MAX, "Light range invalid"

            # Validate scheduling parameters
            assert 0.5 <= self.scheduling.STATIC_DUTY_CYCLE_MINUTES <= 60, "Static duty cycle too extreme"
            assert self.scheduling.ADAPTIVE_MIN_SLEEP_MINUTES < self.scheduling.ADAPTIVE_MAX_SLEEP_MINUTES, "Adaptive sleep range invalid"

            # Validate simulation parameters
            assert 1 <= self.simulation.SIMULATION_DURATION_HOURS <= 168, "Simulation duration too extreme"  # Max 1 week
            assert 0.1 <= self.simulation.TIME_STEP_SECONDS <= 60, "Time step too extreme"

            return True

        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected validation error: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            'power': self.power.__dict__,
            'battery': self.battery.__dict__,
            'sensor': self.sensor.__dict__,
            'scheduling': self.scheduling.__dict__,
            'simulation': self.simulation.__dict__
        }

    def save_to_file(self, filepath: str) -> bool:
        """Save configuration to JSON file"""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file"""
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)

            config = cls()
            if 'power' in data:
                config.power.__dict__.update(data['power'])
            if 'battery' in data:
                config.battery.__dict__.update(data['battery'])
            if 'sensor' in data:
                config.sensor.__dict__.update(data['sensor'])
            if 'scheduling' in data:
                config.scheduling.__dict__.update(data['scheduling'])
            if 'simulation' in data:
                config.simulation.__dict__.update(data['simulation'])

            return config
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return cls()  # Return default configuration on error


# Global configuration instance
config = Config()
