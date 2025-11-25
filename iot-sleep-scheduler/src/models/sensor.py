"""
Sensor Models for IoT Sleep Scheduling System
Realistic sensor simulation with noise, environmental factors, and failure modes
"""

import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

from ..config import config


@dataclass
class SensorReading:
    """Single sensor reading with metadata"""
    timestamp: float
    value: float
    unit: str
    accuracy: float
    is_valid: bool
    raw_value: float
    noise_applied: float = 0.0
    calibration_offset: float = 0.0


@dataclass
class SensorStats:
    """Sensor statistics and performance metrics"""
    total_readings: int = 0
    valid_readings: int = 0
    failed_readings: int = 0
    average_value: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    standard_deviation: float = 0.0
    last_reading_time: float = 0.0
    change_rate: float = 0.0  # Rate of change per time unit
    trend: str = 'stable'  # 'increasing', 'decreasing', 'stable'


class Sensor(ABC):
    """
    Abstract base class for IoT sensors

    Features:
    - Realistic noise modeling
    - Calibration and offset support
    - Failure probability simulation
    - Trend analysis and change detection
    - Historical data tracking
    """

    def __init__(self,
                 name: str,
                 unit: str,
                 accuracy: float,
                 min_value: float,
                 max_value: float,
                 sampling_interval: float = None):
        """
        Initialize sensor with parameters

        Args:
            name: Sensor name identifier
            unit: Unit of measurement
            accuracy: Sensor accuracy (± value)
            min_value: Minimum realistic value
            max_value: Maximum realistic value
            sampling_interval: Time between readings in seconds
        """
        self.name = name
        self.unit = unit
        self.accuracy = accuracy
        self.min_value = min_value
        self.max_value = max_value
        self.sampling_interval = sampling_interval or config.sensor.SAMPLING_INTERVAL_SECONDS

        # Calibration parameters
        self.calibration_offset = 0.0
        self.calibration_factor = 1.0
        self.is_calibrated = True

        # Reading history
        self.reading_history: List[SensorReading] = []
        self.last_reading_time = 0.0
        self.last_value = 0.0

        # Statistics
        self.stats = SensorStats()

        # Environmental factors
        self.temperature_effect = 0.0
        self.humidity_effect = 0.0
        self.aging_factor = 1.0

        # Failure simulation
        self.failure_probability = config.sensor.FAILURE_PROBABILITY
        self.is_failed = False
        self.failure_duration = 0.0

    def read_value(self, current_time: float, environment: Dict[str, float] = None) -> SensorReading:
        """
        Read sensor value with realistic simulation

        Args:
            current_time: Current simulation time
            environment: Environmental conditions affecting sensor

        Returns:
            SensorReading object with value and metadata
        """
        # Check if enough time has passed since last reading
        if current_time - self.last_reading_time < self.sampling_interval:
            return self._get_last_reading()

        self.last_reading_time = current_time
        self.stats.total_readings += 1

        # Simulate sensor failure
        if self._simulate_failure(current_time):
            self.stats.failed_readings += 1
            return self._create_failed_reading(current_time)

        # Generate raw value
        raw_value = self._generate_raw_value(current_time, environment)

        # Apply environmental effects
        env_adjusted_value = self._apply_environmental_effects(raw_value, environment)

        # Apply aging
        aged_value = env_adjusted_value * self.aging_factor

        # Add noise
        noisy_value = self._add_noise(aged_value)

        # Apply calibration
        calibrated_value = (noisy_value + self.calibration_offset) * self.calibration_factor

        # Validate reading range
        final_value = max(self.min_value, min(self.max_value, calibrated_value))

        # Create reading object
        reading = SensorReading(
            timestamp=current_time,
            value=final_value,
            unit=self.unit,
            accuracy=self.accuracy,
            is_valid=True,
            raw_value=raw_value,
            noise_applied=noisy_value - aged_value,
            calibration_offset=self.calibration_offset
        )

        # Update history and statistics
        self._update_history(reading)
        self._update_statistics()

        return reading

    def get_change_rate(self, window_size: int = 5) -> float:
        """
        Calculate rate of change based on recent readings

        Args:
            window_size: Number of recent readings to consider

        Returns:
            Rate of change per time unit
        """
        if len(self.reading_history) < 2:
            return 0.0

        recent_readings = self.reading_history[-window_size:]
        if len(recent_readings) < 2:
            return 0.0

        # Simple linear regression to calculate trend
        n = len(recent_readings)
        times = [r.timestamp for r in recent_readings]
        values = [r.value for r in recent_readings]

        # Calculate slope (rate of change)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(times[i] * values[i] for i in range(n))
        sum_x2 = sum(t * t for t in times)

        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    def detect_trend(self, window_size: int = 5, threshold: float = None) -> str:
        """
        Detect trend in sensor readings

        Args:
            window_size: Number of recent readings to analyze
            threshold: Threshold for trend detection

        Returns:
            Trend string: 'increasing', 'decreasing', 'stable'
        """
        change_rate = self.get_change_rate(window_size)
        threshold = threshold or 0.01

        if abs(change_rate) < threshold:
            return 'stable'
        elif change_rate > 0:
            return 'increasing'
        else:
            return 'decreasing'

    def has_significant_change(self, threshold: float = None) -> bool:
        """
        Check if current reading has significant change from previous

        Args:
            threshold: Change threshold

        Returns:
            True if change is significant
        """
        if threshold is None:
            threshold = self.accuracy * 2  # Default threshold is 2x accuracy

        if len(self.reading_history) < 2:
            return False

        current = self.reading_history[-1].value
        previous = self.reading_history[-2].value
        return abs(current - previous) > threshold

    def calibrate(self, reference_value: float, measured_value: float):
        """
        Calibrate sensor using known reference value

        Args:
            reference_value: True value
            measured_value: Value measured by sensor
        """
        if measured_value != 0:
            self.calibration_factor = reference_value / measured_value
        self.calibration_offset = reference_value - measured_value * self.calibration_factor
        self.is_calibrated = True

    def reset(self):
        """Reset sensor state and statistics"""
        self.reading_history.clear()
        self.stats = SensorStats()
        self.last_reading_time = 0.0
        self.last_value = 0.0
        self.is_failed = False
        self.failure_duration = 0.0

    def _get_last_reading(self) -> SensorReading:
        """Get the last reading if available"""
        if self.reading_history:
            return self.reading_history[-1]
        else:
            # Return default reading
            return SensorReading(
                timestamp=0.0,
                value=0.0,
                unit=self.unit,
                accuracy=self.accuracy,
                is_valid=False,
                raw_value=0.0
            )

    def _simulate_failure(self, current_time: float) -> bool:
        """Simulate random sensor failure"""
        if self.is_failed:
            # Check if failure should recover
            if current_time >= self.failure_duration:
                self.is_failed = False
                return False
            return True

        if random.random() < self.failure_probability:
            self.is_failed = True
            # Random failure duration between 1 and 10 seconds
            self.failure_duration = current_time + random.uniform(1.0, 10.0)
            return True

        return False

    def _create_failed_reading(self, current_time: float) -> SensorReading:
        """Create a failed reading object"""
        return SensorReading(
            timestamp=current_time,
            value=0.0,
            unit=self.unit,
            accuracy=float('inf'),
            is_valid=False,
            raw_value=0.0
        )

    def _add_noise(self, value: float) -> float:
        """Add Gaussian noise to sensor reading"""
        noise_std = self.accuracy / 3.0  # 3-sigma rule
        noise = random.gauss(0, noise_std)
        return value + noise

    def _apply_environmental_effects(self, value: float, environment: Dict[str, float]) -> float:
        """Apply environmental effects to sensor reading"""
        if environment is None:
            return value

        # Temperature effect (default implementation)
        temperature = environment.get('temperature', 25.0)
        temp_effect = (temperature - 25.0) * self.temperature_effect

        # Humidity effect (default implementation)
        humidity = environment.get('humidity', 50.0)
        humidity_effect = (humidity - 50.0) * self.humidity_effect

        return value + temp_effect + humidity_effect

    def _update_history(self, reading: SensorReading):
        """Update reading history"""
        self.reading_history.append(reading)
        self.last_value = reading.value

        # Limit history size
        max_history = 1000
        if len(self.reading_history) > max_history:
            self.reading_history = self.reading_history[-max_history:]

    def _update_statistics(self):
        """Update sensor statistics"""
        if not self.reading_history:
            return

        valid_readings = [r for r in self.reading_history if r.is_valid]
        if not valid_readings:
            return

        values = [r.value for r in valid_readings]

        self.stats.valid_readings = len(valid_readings)
        self.stats.average_value = sum(values) / len(values)
        self.stats.min_value = min(values)
        self.stats.max_value = max(values)

        # Calculate standard deviation
        if len(values) > 1:
            variance = sum((v - self.stats.average_value) ** 2 for v in values) / (len(values) - 1)
            self.stats.standard_deviation = math.sqrt(variance)

        self.stats.last_reading_time = self.reading_history[-1].timestamp
        self.stats.change_rate = self.get_change_rate()
        self.stats.trend = self.detect_trend()

    @abstractmethod
    def _generate_raw_value(self, current_time: float, environment: Dict[str, float]) -> float:
        """Generate realistic raw sensor value - must be implemented by subclasses"""
        pass


class TemperatureSensor(Sensor):
    """Temperature sensor with realistic daily cycles and weather patterns"""

    def __init__(self, sampling_interval: float = None):
        super().__init__(
            name="Temperature",
            unit="°C",
            accuracy=config.sensor.TEMP_ACCURACY,
            min_value=config.sensor.TEMP_MIN,
            max_value=config.sensor.TEMP_MAX,
            sampling_interval=sampling_interval
        )

        # Temperature-specific parameters
        self.base_temperature = 20.0  # Base temperature
        self.daily_amplitude = 10.0   # Daily variation amplitude
        self.weather_offset = 0.0     # Weather effects
        self.next_weather_change = random.uniform(3600, 7200)  # Next weather event
        self.temperature_drift = random.uniform(-0.1, 0.1)  # Long-term drift

        # Environmental sensitivity
        self.temperature_effect = 0.01
        self.humidity_effect = 0.02

    def _generate_raw_value(self, current_time: float, environment: Dict[str, float]) -> float:
        """Generate realistic temperature reading with daily cycles"""
        # Daily temperature cycle (sinusoidal)
        hours_since_midnight = (current_time / 3600) % 24
        daily_cycle = math.sin(2 * math.pi * (hours_since_midnight - 6) / 24)  # Peak at 2 PM

        # Base temperature with daily cycle
        base_temp = self.base_temperature + self.daily_amplitude * daily_cycle

        # Weather events (random changes)
        if current_time >= self.next_weather_change:
            self.weather_offset = random.uniform(-5, 5)
            self.next_weather_change = current_time + random.uniform(3600, 7200)

        # Random fluctuations
        random_variation = random.gauss(0, 0.5)

        # Long-term drift
        time_factor = current_time / (24 * 3600)  # Days
        drift_effect = self.temperature_drift * time_factor

        # Combine all factors
        temperature = base_temp + self.weather_offset + random_variation + drift_effect

        # Apply environmental overrides
        if environment:
            env_temp = environment.get('temperature', None)
            if env_temp is not None:
                # Blend with environmental temperature (80% environment, 20% generated)
                temperature = 0.8 * env_temp + 0.2 * temperature

        return max(self.min_value, min(self.max_value, temperature))


class HumiditySensor(Sensor):
    """Humidity sensor with correlation to temperature and weather patterns"""

    def __init__(self, sampling_interval: float = None):
        super().__init__(
            name="Humidity",
            unit="%",
            accuracy=config.sensor.HUMIDITY_ACCURACY,
            min_value=config.sensor.HUMIDITY_MIN,
            max_value=config.sensor.HUMIDITY_MAX,
            sampling_interval=sampling_interval
        )

        # Humidity-specific parameters
        self.base_humidity = 50.0
        self.daily_amplitude = 15.0
        self.weather_effect = 0.0
        self.next_weather_change = random.uniform(3600, 7200)
        self.humidity_drift = random.uniform(-0.5, 0.5)

        # Environmental sensitivity
        self.temperature_effect = -0.3  # Inverse relationship with temperature
        self.humidity_effect = 0.1

    def _generate_raw_value(self, current_time: float, environment: Dict[str, float]) -> float:
        """Generate realistic humidity reading with temperature correlation"""
        # Daily humidity cycle (inverse of temperature)
        hours_since_midnight = (current_time / 3600) % 24
        daily_cycle = -math.sin(2 * math.pi * (hours_since_midnight - 6) / 24)  # Peak humidity at 6 AM

        # Base humidity with daily cycle
        base_humidity = self.base_humidity + self.daily_amplitude * daily_cycle

        # Weather events (rain, dry conditions)
        if current_time >= self.next_weather_change:
            self.weather_effect = random.uniform(-20, 20)
            self.next_weather_change = current_time + random.uniform(3600, 7200)

        # Random fluctuations
        random_variation = random.gauss(0, 2.0)

        # Long-term drift
        time_factor = current_time / (24 * 3600)  # Days
        drift_effect = self.humidity_drift * time_factor

        # Temperature correlation
        temp_effect = 0
        if environment and 'temperature' in environment:
            temp = environment['temperature']
            # Higher temperature typically means lower humidity
            temp_effect = (temp - 25.0) * self.temperature_effect

        # Combine all factors
        humidity = base_humidity + self.weather_effect + random_variation + drift_effect + temp_effect

        # Apply environmental overrides
        if environment and 'humidity' in environment:
            env_humidity = environment['humidity']
            # Blend with environmental humidity (70% environment, 30% generated)
            humidity = 0.7 * env_humidity + 0.3 * humidity

        return max(self.min_value, min(self.max_value, humidity))


class LightSensor(Sensor):
    """Light sensor with day/night cycles and weather effects"""

    def __init__(self, sampling_interval: float = None):
        super().__init__(
            name="Light",
            unit="lux",
            accuracy=config.sensor.LIGHT_ACCURACY,
            min_value=config.sensor.LIGHT_MIN,
            max_value=config.sensor.LIGHT_MAX,
            sampling_interval=sampling_interval
        )

        # Light-specific parameters
        self.max_daylight = 800.0  # Maximum daylight lux
        self.night_level = 5.0     # Night-time ambient light
        self.weather_factor = 1.0  # Cloud cover effect
        self.next_weather_change = random.uniform(1800, 3600)  # Weather changes more frequently
        self.season_factor = 1.0   # Seasonal variation

        # Environmental sensitivity
        self.temperature_effect = 0.05
        self.humidity_effect = 0.02

    def _generate_raw_value(self, current_time: float, environment: Dict[str, float]) -> float:
        """Generate realistic light reading with day/night cycles"""
        # Time of day
        hours_since_midnight = (current_time / 3600) % 24

        # Calculate sun position (simplified)
        if 6 <= hours_since_midnight <= 18:  # Daylight hours
            # Sun elevation angle (simplified)
            sun_progress = (hours_since_midnight - 6) / 12  # 0 to 1 during daylight
            sun_elevation = math.sin(sun_progress * math.pi)  # Peak at noon

            # Base light level from sun
            base_light = self.night_level + (self.max_daylight - self.night_level) * sun_elevation
        else:
            # Night time
            base_light = self.night_level

        # Weather effects (cloud cover)
        if current_time >= self.next_weather_change:
            self.weather_factor = random.uniform(0.1, 1.0)  # 0.1 = heavy clouds, 1.0 = clear sky
            self.next_weather_change = current_time + random.uniform(1800, 3600)

        # Apply weather factor
        weather_adjusted_light = base_light * self.weather_factor

        # Random fluctuations (sensor noise, small shadows)
        random_variation = random.gauss(0, 10.0)

        # Combine all factors
        light_level = weather_adjusted_light + random_variation

        # Apply environmental overrides
        if environment and 'light' in environment:
            env_light = environment['light']
            # Blend with environmental light (60% environment, 40% generated)
            light_level = 0.6 * env_light + 0.4 * light_level

        return max(self.min_value, min(self.max_value, light_level))


class SensorManager:
    """Manager for multiple sensors with coordinated reading and data analysis"""

    def __init__(self):
        self.sensors: Dict[str, Sensor] = {}
        self.reading_intervals = {}

    def add_sensor(self, sensor: Sensor):
        """Add a sensor to the manager"""
        self.sensors[sensor.name] = sensor
        self.reading_intervals[sensor.name] = sensor.sampling_interval

    def remove_sensor(self, sensor_name: str):
        """Remove a sensor from the manager"""
        if sensor_name in self.sensors:
            del self.sensors[sensor_name]
            del self.reading_intervals[sensor_name]

    def read_all_sensors(self, current_time: float, environment: Dict[str, float] = None) -> Dict[str, SensorReading]:
        """Read all sensors and return dictionary of readings"""
        readings = {}
        for name, sensor in self.sensors.items():
            readings[name] = sensor.read_value(current_time, environment)
        return readings

    def get_sensor_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all sensors"""
        summary = {}
        for name, sensor in self.sensors.items():
            summary[name] = {
                'name': sensor.name,
                'unit': sensor.unit,
                'accuracy': sensor.accuracy,
                'total_readings': sensor.stats.total_readings,
                'valid_readings': sensor.stats.valid_readings,
                'failed_readings': sensor.stats.failed_readings,
                'average_value': sensor.stats.average_value,
                'min_value': sensor.stats.min_value,
                'max_value': sensor.stats.max_value,
                'standard_deviation': sensor.stats.standard_deviation,
                'current_trend': sensor.stats.trend,
                'change_rate': sensor.stats.change_rate,
                'is_calibrated': sensor.is_calibrated,
                'last_reading': sensor._get_last_reading().value if sensor.reading_history else 0.0
            }
        return summary

    def detect_significant_changes(self, thresholds: Dict[str, float] = None) -> Dict[str, bool]:
        """Check for significant changes across all sensors"""
        if thresholds is None:
            thresholds = {
                'Temperature': config.sensor.TEMP_CHANGE_THRESHOLD,
                'Humidity': config.sensor.HUMIDITY_CHANGE_THRESHOLD,
                'Light': config.sensor.LIGHT_CHANGE_THRESHOLD
            }

        changes = {}
        for name, sensor in self.sensors.items():
            threshold = thresholds.get(name, sensor.accuracy * 2)
            changes[name] = sensor.has_significant_change(threshold)

        return changes

    def get_overall_activity_level(self) -> float:
        """Calculate overall activity level based on all sensors"""
        if not self.sensors:
            return 0.0

        total_change_rate = sum(abs(sensor.get_change_rate()) for sensor in self.sensors.values())
        return total_change_rate / len(self.sensors)

    def reset_all_sensors(self):
        """Reset all sensors to initial state"""
        for sensor in self.sensors.values():
            sensor.reset()