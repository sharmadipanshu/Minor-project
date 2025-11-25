"""
Battery Model for IoT Sleep Scheduling System
Realistic battery simulation with discharge characteristics, temperature effects, and lifetime estimation
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class BatteryState:
    """Battery state information at a specific point in time"""
    timestamp: float
    charge_percentage: float
    voltage: float
    temperature_celsius: float
    current_draw_ma: float
    energy_consumed_mah: float
    cycle_count: int = 0
    is_charging: bool = False


@dataclass
class BatteryMetrics:
    """Battery performance metrics and statistics"""
    total_energy_consumed_mah: float = 0.0
    average_current_ma: float = 0.0
    peak_current_ma: float = 0.0
    time_in_critical_battery: float = 0.0
    discharge_rate_mah_per_hour: float = 0.0
    estimated_remaining_hours: float = 0.0
    efficiency_factor: float = 1.0
    temperature_adjustment_factor: float = 1.0


class Battery:
    """
    Realistic battery model for IoT sensor nodes

    Features:
    - Non-linear discharge curve based on state of charge
    - Temperature effects on capacity and performance
    - Self-discharge modeling
    - Cycle counting for lifetime estimation
    - Voltage-based energy calculations
    - Realistic Li-ion battery behavior
    """

    def __init__(self,
                 capacity_mah: float = None,
                 initial_charge_percentage: float = None,
                 initial_temperature: float = 25.0):
        """
        Initialize battery with specified parameters

        Args:
            capacity_mah: Battery capacity in milliampere-hours
            initial_charge_percentage: Initial charge level (0-100)
            initial_temperature: Initial battery temperature in Celsius
        """
        # Configuration parameters
        self.capacity_mah = capacity_mah or config.battery.BATTERY_CAPACITY_MAH
        self.initial_charge_percentage = initial_charge_percentage or config.battery.INITIAL_CHARGE_PERCENTAGE

        # Battery state
        self.current_charge_mah = (self.initial_charge_percentage / 100.0) * self.capacity_mah
        self.temperature_celsius = initial_temperature
        self.cycle_count = 0
        self.total_energy_consumed_mah = 0.0

        # State tracking
        self.is_charging = False
        self.charge_history: List[BatteryState] = []
        self.current_draw_ma = 0.0
        self.last_update_time = 0.0

        # Performance metrics
        self.metrics = BatteryMetrics()
        self.peak_current_ma = 0.0
        self.cumulative_current_ma_seconds = 0.0
        self.total_time_seconds = 0.0

        # Temperature effects
        self.temperature_history: List[Tuple[float, float]] = []  # (timestamp, temperature)
        self.capacity_degradation_factor = 1.0  # Tracks capacity loss over cycles

        # Initialize charge history
        self._record_state(0.0)

    def get_remaining_percentage(self) -> float:
        """
        Get remaining battery charge as percentage

        Returns:
            Battery charge percentage (0-100)
        """
        return min(100.0, max(0.0, (self.current_charge_mah / self.capacity_mah) * 100.0))

    def get_remaining_capacity_mah(self) -> float:
        """
        Get remaining battery capacity in mAh

        Returns:
            Remaining capacity in milliampere-hours
        """
        return max(0.0, self.current_charge_mah)

    def get_voltage(self) -> float:
        """
        Get battery voltage based on state of charge and temperature

        Returns:
            Battery voltage in volts
        """
        percentage = self.get_remaining_percentage()

        # Interpolate voltage from discharge curve
        voltage = self._interpolate_voltage(percentage)

        # Apply temperature effects
        voltage += self._get_temperature_voltage_offset()

        # Apply degradation effects
        voltage *= self.capacity_degradation_factor

        return max(0.0, voltage)

    def consume_current(self, current_ma: float, duration_seconds: float) -> float:
        """
        Consume current for specified duration

        Args:
            current_ma: Current draw in milliamps
            duration_seconds: Duration in seconds

        Returns:
            Energy consumed in mAh
        """
        if current_ma <= 0 or duration_seconds <= 0:
            return 0.0

        # Calculate energy consumption (mAh = mA * hours)
        duration_hours = duration_seconds / 3600.0
        energy_consumed_mah = current_ma * duration_hours

        # Apply efficiency factor
        energy_consumed_mah /= config.battery.EFFICIENCY_FACTOR

        # Apply temperature effects
        temp_factor = self._get_temperature_efficiency_factor()
        energy_consumed_mah *= temp_factor

        # Update battery charge
        self.current_charge_mah = max(0.0, self.current_charge_mah - energy_consumed_mah)
        self.total_energy_consumed_mah += energy_consumed_mah

        # Update metrics
        self._update_metrics(current_ma, duration_seconds)

        # Record state
        self.current_draw_ma = current_ma
        self.peak_current_ma = max(self.peak_current_ma, current_ma)

        return energy_consumed_mah

    def update(self, current_time: float, environment_temperature: float = None):
        """
        Update battery state for current time step

        Args:
            current_time: Current simulation time in seconds
            environment_temperature: Ambient temperature in Celsius
        """
        if current_time <= self.last_update_time:
            return

        # Calculate time delta
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update temperature (with thermal inertia)
        if environment_temperature is not None:
            self._update_temperature(environment_temperature, dt)

        # Apply self-discharge
        self._apply_self_discharge(dt)

        # Record state periodically
        if len(self.charge_history) == 0 or \
           current_time - self.charge_history[-1].timestamp >= config.simulation.LOG_INTERVAL_SECONDS:
            self._record_state(current_time)

        # Update metrics
        self.total_time_seconds += dt
        self.metrics.total_energy_consumed_mah = self.total_energy_consumed_mah
        self.metrics.peak_current_ma = self.peak_current_ma

        # Calculate average current
        if self.total_time_seconds > 0:
            self.metrics.average_current_ma = (self.cumulative_current_ma_seconds / self.total_time_seconds) * 3600.0

        # Update discharge rate
        if len(self.charge_history) >= 2:
            self._update_discharge_rate()

        # Estimate remaining time
        self._estimate_remaining_time()

    def reset(self, initial_charge_percentage: float = None):
        """
        Reset battery to full charge or specified level

        Args:
            initial_charge_percentage: Optional initial charge percentage
        """
        if initial_charge_percentage is None:
            initial_charge_percentage = config.battery.INITIAL_CHARGE_PERCENTAGE

        self.current_charge_mah = (initial_charge_percentage / 100.0) * self.capacity_mah
        self.total_energy_consumed_mah = 0.0
        self.cycle_count = 0
        self.peak_current_ma = 0.0
        self.cumulative_current_ma_seconds = 0.0
        self.total_time_seconds = 0.0
        self.charge_history.clear()
        self.temperature_history.clear()
        self.capacity_degradation_factor = 1.0
        self.last_update_time = 0.0
        self.metrics = BatteryMetrics()

        self._record_state(0.0)

    def is_depleted(self) -> bool:
        """
        Check if battery is below critical threshold

        Returns:
            True if battery is depleted, False otherwise
        """
        return self.get_remaining_percentage() <= config.battery.CRITICAL_BATTERY_PERCENTAGE

    def is_low(self) -> bool:
        """
        Check if battery is below low threshold

        Returns:
            True if battery is low, False otherwise
        """
        return self.get_remaining_percentage() <= config.battery.LOW_BATTERY_THRESHOLD

    def get_state_of_charge_health(self) -> float:
        """
        Get battery health based on cycle count and degradation

        Returns:
            Battery health percentage (0-100)
        """
        # Simple degradation model: 5% capacity loss per 500 cycles
        cycle_degradation = min(0.5, self.cycle_count / 500.0 * 0.05)

        # Temperature degradation
        temp_degradation = 0.0
        for _, temp in self.temperature_history:
            if temp < 0 or temp > 45:  # Extreme temperatures
                temp_degradation += 0.001

        health = 1.0 - cycle_degradation - temp_degradation
        return max(0.0, min(100.0, health * 100.0))

    def _interpolate_voltage(self, percentage: float) -> float:
        """
        Interpolate voltage from discharge curve

        Args:
            percentage: State of charge percentage

        Returns:
            Voltage in volts
        """
        curve = config.battery.DISCHARGE_CURVE

        if percentage >= 100.0:
            return curve[100.0]
        if percentage <= 0.0:
            return curve[0.0]

        # Find surrounding points
        percentages = sorted(curve.keys())

        for i in range(len(percentages) - 1):
            if percentages[i] <= percentage <= percentages[i + 1]:
                # Linear interpolation
                x1, y1 = percentages[i], curve[percentages[i]]
                x2, y2 = percentages[i + 1], curve[percentages[i + 1]]

                if x2 - x1 == 0:
                    return y1

                slope = (y2 - y1) / (x2 - x1)
                return y1 + slope * (percentage - x1)

        return curve[percentages[0]]

    def _get_temperature_voltage_offset(self) -> float:
        """
        Calculate voltage offset due to temperature

        Returns:
            Voltage offset in volts
        """
        # Li-ion voltage varies with temperature
        if self.temperature_celsius < 0:
            return -0.1  # Cold reduces voltage
        elif self.temperature_celsius > 40:
            return -0.05  # Heat reduces voltage slightly
        else:
            return 0.0

    def _get_temperature_efficiency_factor(self) -> float:
        """
        Calculate efficiency factor based on temperature

        Returns:
            Efficiency multiplier (0.8-1.2)
        """
        # Optimal temperature range: 20-30°C
        if 20 <= self.temperature_celsius <= 30:
            return 1.0
        elif self.temperature_celsius < 0 or self.temperature_celsius > 45:
            return 1.2  # Reduced efficiency at extreme temperatures
        else:
            return 1.1  # Slightly reduced efficiency

    def _update_temperature(self, ambient_temp: float, dt: float):
        """
        Update battery temperature with thermal inertia

        Args:
            ambient_temp: Ambient temperature in Celsius
            dt: Time delta in seconds
        """
        # Simple thermal model with time constant
        thermal_time_constant = 300.0  # 5 minutes
        alpha = dt / thermal_time_constant

        self.temperature_celsius += alpha * (ambient_temp - self.temperature_celsius)
        self.temperature_history.append((self.last_update_time, self.temperature_celsius))

        # Keep only recent history
        max_history = 1000
        if len(self.temperature_history) > max_history:
            self.temperature_history = self.temperature_history[-max_history:]

    def _apply_self_discharge(self, dt: float):
        """
        Apply self-discharge to battery

        Args:
            dt: Time delta in seconds
        """
        # Self-discharge rate per month converted to per second
        monthly_rate = config.battery.SELF_DISCHARGE_RATE_PER_MONTH / 100.0
        seconds_per_month = 30.0 * 24.0 * 3600.0
        second_rate = monthly_rate / seconds_per_month

        # Apply self-discharge
        self_discharge_mah = self.capacity_mah * second_rate * dt
        self.current_charge_mah = max(0.0, self.current_charge_mah - self_discharge_mah)

    def _update_metrics(self, current_ma: float, duration_seconds: float):
        """
        Update performance metrics

        Args:
            current_ma: Current draw in milliamps
            duration_seconds: Duration in seconds
        """
        self.cumulative_current_ma_seconds += current_ma * duration_seconds

        # Update average current
        if self.total_time_seconds > 0:
            self.metrics.average_current_ma = (self.cumulative_current_ma_seconds / self.total_time_seconds) * 3600.0

        # Track time in critical battery
        if self.is_depleted():
            self.metrics.time_in_critical_battery += duration_seconds

    def _update_discharge_rate(self):
        """
        Update current discharge rate based on recent history
        """
        if len(self.charge_history) < 2:
            return

        # Get last two states
        recent = self.charge_history[-1]
        previous = self.charge_history[-2]

        # Calculate discharge rate
        time_diff = recent.timestamp - previous.timestamp
        if time_diff > 0:
            charge_diff = previous.charge_percentage - recent.charge_percentage
            time_hours = time_diff / 3600.0
            self.metrics.discharge_rate_mah_per_hour = (charge_diff / 100.0) * self.capacity_mah / time_hours

    def _estimate_remaining_time(self):
        """
        Estimate remaining operation time based on current consumption
        """
        if self.metrics.average_current_ma <= 0:
            self.metrics.estimated_remaining_hours = float('inf')
            return

        remaining_mah = self.get_remaining_capacity_mah()
        self.metrics.estimated_remaining_hours = remaining_mah / self.metrics.average_current_ma

    def _record_state(self, timestamp: float):
        """
        Record current battery state

        Args:
            timestamp: Current simulation time
        """
        state = BatteryState(
            timestamp=timestamp,
            charge_percentage=self.get_remaining_percentage(),
            voltage=self.get_voltage(),
            temperature_celsius=self.temperature_celsius,
            current_draw_ma=self.current_draw_ma,
            energy_consumed_mah=self.total_energy_consumed_mah,
            cycle_count=self.cycle_count,
            is_charging=self.is_charging
        )
        self.charge_history.append(state)

        # Limit history size
        max_history = 10000
        if len(self.charge_history) > max_history:
            self.charge_history = self.charge_history[-max_history:]

    def get_summary(self) -> Dict:
        """
        Get battery summary statistics

        Returns:
            Dictionary containing battery summary
        """
        return {
            'capacity_mah': self.capacity_mah,
            'current_charge_percentage': self.get_remaining_percentage(),
            'current_charge_mah': self.get_remaining_capacity_mah(),
            'voltage': self.get_voltage(),
            'temperature_celsius': self.temperature_celsius,
            'total_energy_consumed_mah': self.total_energy_consumed_mah,
            'cycle_count': self.cycle_count,
            'health_percentage': self.get_state_of_charge_health(),
            'is_depleted': self.is_depleted(),
            'is_low': self.is_low(),
            'estimated_remaining_hours': self.metrics.estimated_remaining_hours,
            'average_current_ma': self.metrics.average_current_ma,
            'peak_current_ma': self.metrics.peak_current_ma
        }