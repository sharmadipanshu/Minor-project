"""
Power Model for IoT Sleep Scheduling System
Realistic power consumption calculations and energy estimation for different operational modes
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

from ..config import config


class PowerState(Enum):
    """Power consumption states for IoT nodes"""
    ACTIVE = "active"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    TRANSMITTING = "transmitting"
    SENSING = "sensing"
    PROCESSING = "processing"


@dataclass
class PowerConsumption:
    """Power consumption data for a specific state"""
    state: PowerState
    current_ma: float
    voltage: float
    power_mw: float
    duration_seconds: float
    energy_mah: float
    energy_mwh: float


@dataclass
class PowerProfile:
    """Complete power consumption profile for a time period"""
    total_energy_mah: float = 0.0
    total_energy_mwh: float = 0.0
    average_power_mw: float = 0.0
    peak_power_mw: float = 0.0
    state_distribution: Dict[PowerState, float] = field(default_factory=dict)
    consumption_history: List[PowerConsumption] = field(default_factory=list)
    total_duration_seconds: float = 0.0


class PowerModel:
    """
    Comprehensive power consumption model for IoT sensor nodes

    Features:
    - State-specific power consumption calculations
    - Voltage-based energy estimation
    - Transmission overhead modeling
    - Battery-aware power management
    - Lifetime estimation and prediction
    - Temperature effects on power consumption
    """

    def __init__(self, voltage: float = None):
        """
        Initialize power model with operating voltage

        Args:
            voltage: Operating voltage in volts (default from config)
        """
        self.voltage = voltage or config.power.VOLTAGE

        # State-specific current consumption (from config)
        self.state_currents = {
            PowerState.ACTIVE: config.power.ACTIVE_CURRENT_MA,
            PowerState.LIGHT_SLEEP: config.power.SLEEP_CURRENT_MA,
            PowerState.DEEP_SLEEP: config.power.DEEP_SLEEP_CURRENT_MA,
            PowerState.TRANSMITTING: config.power.TX_CURRENT_MA,
            PowerState.SENSING: 2.0,  # Additional current for sensing
            PowerState.PROCESSING: 8.0  # Additional current for data processing
        }

        # Transmission parameters
        self.tx_duration = config.power.TX_DURATION_SECONDS
        self.tx_pre_wake_energy = config.power.TX_PRE_WAKE_ENERGY_MAH

        # Power consumption tracking
        self.current_profile = PowerProfile()
        self.historical_profiles: List[PowerProfile] = []

        # Temperature effects
        self.temperature_coefficient = 0.002  # 0.2% change per degree Celsius
        self.reference_temperature = 25.0  # Reference temperature for calculations

        # Efficiency factors
        self.voltage_regulator_efficiency = 0.85  # DC-DC converter efficiency
        self.power_management_overhead = 0.05  # 5% overhead for power management

    def get_current_draw(self, state: PowerState, temperature: float = 25.0) -> float:
        """
        Get current draw for a specific power state with temperature compensation

        Args:
            state: Power state
            temperature: Temperature in Celsius

        Returns:
            Current draw in milliamps
        """
        base_current = self.state_currents.get(state, 0.0)

        # Apply temperature effects
        temp_factor = 1.0 + self.temperature_coefficient * (temperature - self.reference_temperature)
        temp_adjusted_current = base_current * temp_factor

        # Apply efficiency factors
        efficiency_factor = 1.0 / (self.voltage_regulator_efficiency * (1.0 - self.power_management_overhead))
        total_current = temp_adjusted_current * efficiency_factor

        return total_current

    def calculate_power_consumption(self, state: PowerState, duration_seconds: float,
                                   temperature: float = 25.0, voltage: float = None) -> PowerConsumption:
        """
        Calculate power consumption for a specific state and duration

        Args:
            state: Power state
            duration_seconds: Duration in seconds
            temperature: Temperature in Celsius
            voltage: Operating voltage (override default)

        Returns:
            PowerConsumption object with detailed calculations
        """
        operating_voltage = voltage or self.voltage
        current_ma = self.get_current_draw(state, temperature)

        # Calculate power in milliwatts
        power_mw = current_ma * operating_voltage

        # Calculate energy in milliampere-hours
        duration_hours = duration_seconds / 3600.0
        energy_mah = current_ma * duration_hours

        # Calculate energy in milliwatt-hours
        energy_mwh = power_mw * duration_hours

        consumption = PowerConsumption(
            state=state,
            current_ma=current_ma,
            voltage=operating_voltage,
            power_mw=power_mw,
            duration_seconds=duration_seconds,
            energy_mah=energy_mah,
            energy_mwh=energy_mwh
        )

        return consumption

    def calculate_transmission_energy(self, data_size_bytes: int = 100,
                                     temperature: float = 25.0) -> float:
        """
        Calculate energy consumption for data transmission

        Args:
            data_size_bytes: Size of data to transmit
            temperature: Temperature in Celsius

        Returns:
            Total transmission energy in mAh
        """
        # Pre-transmission wake-up energy
        total_energy = self.tx_pre_wake_energy

        # Main transmission energy
        tx_consumption = self.calculate_power_consumption(
            PowerState.TRANSMITTING, self.tx_duration, temperature
        )
        total_energy += tx_consumption.energy_mah

        # Adjust for data size (longer transmission for larger data)
        if data_size_bytes > 100:
            size_factor = data_size_bytes / 100.0
            total_energy *= size_factor

        return total_energy

    def calculate_sensing_energy(self, num_sensors: int = 3,
                               temperature: float = 25.0) -> float:
        """
        Calculate energy consumption for sensor readings

        Args:
            num_sensors: Number of sensors to read
            temperature: Temperature in Celsius

        Returns:
            Total sensing energy in mAh
        """
        # Energy per sensor reading (including ADC conversion)
        sensing_duration_per_sensor = 0.01  # 10ms per sensor
        total_sensing_duration = num_sensors * sensing_duration_per_sensor

        sensing_consumption = self.calculate_power_consumption(
            PowerState.SENSING, total_sensing_duration, temperature
        )

        # Add base sensor power from config
        base_sensor_energy = num_sensors * config.sensor.SENSOR_POWER_CONSUMPTION_MAH

        return sensing_consumption.energy_mah + base_sensor_energy

    def calculate_processing_energy(self, processing_time_seconds: float = 0.1,
                                  temperature: float = 25.0) -> float:
        """
        Calculate energy consumption for data processing

        Args:
            processing_time_seconds: Time spent processing data
            temperature: Temperature in Celsius

        Returns:
            Processing energy in mAh
        """
        processing_consumption = self.calculate_power_consumption(
            PowerState.PROCESSING, processing_time_seconds, temperature
        )

        return processing_consumption.energy_mah

    def estimate_lifetime(self, battery_capacity_mah: float, consumption_rate_mah_per_hour: float,
                         safety_margin: float = 0.1) -> Dict[str, float]:
        """
        Estimate battery lifetime based on consumption rate

        Args:
            battery_capacity_mah: Battery capacity in mAh
            consumption_rate_mah_per_hour: Current consumption rate
            safety_margin: Safety margin fraction (0.1 = 10%)

        Returns:
            Dictionary with lifetime estimates
        """
        if consumption_rate_mah_per_hour <= 0:
            return {
                'hours': float('inf'),
                'days': float('inf'),
                'weeks': float('inf'),
                'months': float('inf'),
                'years': float('inf')
            }

        # Apply safety margin
        usable_capacity = battery_capacity_mah * (1.0 - safety_margin)

        # Calculate lifetime
        lifetime_hours = usable_capacity / consumption_rate_mah_per_hour

        return {
            'hours': lifetime_hours,
            'days': lifetime_hours / 24.0,
            'weeks': lifetime_hours / (24.0 * 7.0),
            'months': lifetime_hours / (24.0 * 30.0),
            'years': lifetime_hours / (24.0 * 365.0)
        }

    def create_duty_cycle_profile(self, active_time_percent: float,
                                 sleep_time_percent: float, deep_sleep_time_percent: float,
                                 transmission_time_percent: float, total_duration_hours: float) -> PowerProfile:
        """
        Create power profile based on duty cycle percentages

        Args:
            active_time_percent: Percentage of time in active mode
            sleep_time_percent: Percentage of time in light sleep
            deep_sleep_time_percent: Percentage of time in deep sleep
            transmission_time_percent: Percentage of time transmitting
            total_duration_hours: Total duration in hours

        Returns:
            PowerProfile for the duty cycle
        """
        profile = PowerProfile()
        profile.total_duration_seconds = total_duration_hours * 3600.0

        # Calculate time in each state
        total_seconds = profile.total_duration_seconds
        active_seconds = total_seconds * (active_time_percent / 100.0)
        sleep_seconds = total_seconds * (sleep_time_percent / 100.0)
        deep_sleep_seconds = total_seconds * (deep_sleep_time_percent / 100.0)
        tx_seconds = total_seconds * (transmission_time_percent / 100.0)

        # Calculate consumption for each state
        consumptions = [
            self.calculate_power_consumption(PowerState.ACTIVE, active_seconds),
            self.calculate_power_consumption(PowerState.LIGHT_SLEEP, sleep_seconds),
            self.calculate_power_consumption(PowerState.DEEP_SLEEP, deep_sleep_seconds),
            self.calculate_power_consumption(PowerState.TRANSMITTING, tx_seconds)
        ]

        # Aggregate results
        total_energy_mah = sum(c.energy_mah for c in consumptions)
        total_energy_mwh = sum(c.energy_mwh for c in consumptions)
        total_power_mw = sum(c.power_mw * c.duration_seconds for c in consumptions) / total_seconds

        profile.total_energy_mah = total_energy_mah
        profile.total_energy_mwh = total_energy_mwh
        profile.average_power_mw = total_power_mw
        profile.peak_power_mw = max(c.power_mw for c in consumptions)
        profile.consumption_history = consumptions

        # Calculate state distribution
        profile.state_distribution = {
            PowerState.ACTIVE: active_time_percent,
            PowerState.LIGHT_SLEEP: sleep_time_percent,
            PowerState.DEEP_SLEEP: deep_sleep_time_percent,
            PowerState.TRANSMITTING: transmission_time_percent
        }

        return profile

    def compare_power_consumption(self, profile1: PowerProfile, profile2: PowerProfile) -> Dict[str, float]:
        """
        Compare two power consumption profiles

        Args:
            profile1: First power profile
            profile2: Second power profile

        Returns:
            Dictionary with comparison metrics
        """
        if profile1.total_duration_seconds == 0 or profile2.total_duration_seconds == 0:
            return {'improvement_percentage': 0.0, 'energy_savings_mah': 0.0}

        # Normalize to same duration
        duration_factor = profile2.total_duration_seconds / profile1.total_duration_seconds
        normalized_energy1 = profile1.total_energy_mah * duration_factor

        energy_difference = normalized_energy1 - profile2.total_energy_mah
        improvement_percentage = (energy_difference / normalized_energy1) * 100.0 if normalized_energy1 > 0 else 0.0

        return {
            'energy_difference_mah': energy_difference,
            'improvement_percentage': improvement_percentage,
            'energy_savings_mah': max(0.0, energy_difference),
            'profile1_avg_power_mw': profile1.average_power_mw,
            'profile2_avg_power_mw': profile2.average_power_mw,
            'power_reduction_mw': profile1.average_power_mw - profile2.average_power_mw
        }

    def analyze_power_trends(self, profiles: List[PowerProfile]) -> Dict[str, float]:
        """
        Analyze power consumption trends over multiple profiles

        Args:
            profiles: List of power profiles to analyze

        Returns:
            Dictionary with trend analysis
        """
        if len(profiles) < 2:
            return {
                'trend': 'insufficient_data',
                'average_consumption_mah': 0.0,
                'consumption_variance': 0.0,
                'peak_consumption_mah': 0.0,
                'min_consumption_mah': 0.0
            }

        consumptions = [p.total_energy_mah for p in profiles]

        avg_consumption = sum(consumptions) / len(consumptions)
        variance = sum((c - avg_consumption) ** 2 for c in consumptions) / len(consumptions)
        peak_consumption = max(consumptions)
        min_consumption = min(consumptions)

        # Simple trend detection
        if len(consumptions) >= 3:
            recent = consumptions[-3:]
            if recent[-1] > recent[0] and recent[-1] > recent[1]:
                trend = 'increasing'
            elif recent[-1] < recent[0] and recent[-1] < recent[1]:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'average_consumption_mah': avg_consumption,
            'consumption_variance': variance,
            'consumption_std': math.sqrt(variance),
            'peak_consumption_mah': peak_consumption,
            'min_consumption_mah': min_consumption,
            'consumption_range_mah': peak_consumption - min_consumption
        }

    def get_power_state_breakdown(self, profile: PowerProfile) -> Dict[str, Dict[str, float]]:
        """
        Get detailed breakdown of power consumption by state

        Args:
            profile: Power profile to analyze

        Returns:
            Dictionary with detailed state breakdown
        """
        breakdown = {}
        total_energy = profile.total_energy_mah

        for state, percentage in profile.state_distribution.items():
            state_energy = total_energy * (percentage / 100.0)
            percentage_of_total = (state_energy / total_energy * 100.0) if total_energy > 0 else 0.0

            breakdown[state.value] = {
                'energy_mah': state_energy,
                'energy_mwh': state_energy * self.voltage,
                'percentage_time': percentage,
                'percentage_energy': percentage_of_total,
                'current_ma': self.state_currents.get(state, 0.0),
                'power_mw': self.state_currents.get(state, 0.0) * self.voltage
            }

        return breakdown

    def get_optimal_sleep_duration(self, battery_percentage: float, desired_lifetime_days: float) -> float:
        """
        Calculate optimal sleep duration to achieve desired lifetime

        Args:
            battery_percentage: Current battery percentage
            desired_lifetime_days: Desired operational lifetime in days

        Returns:
            Recommended sleep duration in minutes
        """
        if battery_percentage <= 0:
            return config.scheduling.ADAPTIVE_MAX_SLEEP_MINUTES

        # Calculate required average power consumption
        remaining_capacity_ah = (config.battery.BATTERY_CAPACITY_MAH / 1000.0) * (battery_percentage / 100.0)
        required_current_a = remaining_capacity_ah / (desired_lifetime_days * 24.0)
        required_current_ma = required_current_a * 1000.0

        # Estimate active time impact
        active_current_ma = self.state_currents[PowerState.ACTIVE]
        sleep_current_ma = self.state_currents[PowerState.LIGHT_SLEEP]
        deep_sleep_current_ma = self.state_currents[PowerState.DEEP_SLEEP]

        # Assume 5% active time, calculate required sleep time
        active_fraction = 0.05
        required_sleep_fraction = (required_current_ma - active_current_ma * active_fraction) / \
                                (deep_sleep_current_ma - required_current_ma)

        if required_sleep_fraction < 0:
            return config.scheduling.ADAPTIVE_MIN_SLEEP_MINUTES
        elif required_sleep_fraction > 0.95:
            return config.scheduling.ADAPTIVE_MAX_SLEEP_MINUTES
        else:
            # Convert fraction to sleep duration
            sleep_duration = (required_sleep_fraction / (1 - active_fraction)) * 60  # Convert to minutes
            return max(config.scheduling.ADAPTIVE_MIN_SLEEP_MINUTES,
                       min(config.scheduling.ADAPTIVE_MAX_SLEEP_MINUTES, sleep_duration))

    def reset_profile(self):
        """Reset current power profile"""
        self.current_profile = PowerProfile()

    def save_profile(self, profile: PowerProfile = None):
        """Save current or specified profile to history"""
        if profile is None:
            profile = self.current_profile
        self.historical_profiles.append(profile.copy())

    def get_summary(self) -> Dict:
        """
        Get comprehensive power model summary

        Returns:
            Dictionary containing power model summary
        """
        return {
            'voltage': self.voltage,
            'state_currents': {state.value: current for state, current in self.state_currents.items()},
            'tx_duration_seconds': self.tx_duration,
            'voltage_regulator_efficiency': self.voltage_regulator_efficiency,
            'temperature_coefficient': self.temperature_coefficient,
            'current_profile': {
                'total_energy_mah': self.current_profile.total_energy_mah,
                'total_energy_mwh': self.current_profile.total_energy_mwh,
                'average_power_mw': self.current_profile.average_power_mw,
                'peak_power_mw': self.current_profile.peak_power_mw,
                'state_distribution': {state.value: pct for state, pct in self.current_profile.state_distribution.items()}
            },
            'historical_profiles_count': len(self.historical_profiles)
        }