"""
Sleep Scheduling Algorithms for IoT Power Optimization
Implementation of static and adaptive duty cycling algorithms with intelligent wake-up triggers
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from ..config import config


class WakeupReason(Enum):
    """Reasons for node wake-up"""
    SCHEDULED_TIMER = "scheduled_timer"
    THRESHOLD_BREACH = "threshold_breach"
    HIGH_ACTIVITY = "high_activity"
    BATTERY_CONSERVATIVE = "battery_conservative"
    BATTERY_CRITICAL = "battery_critical"
    EXTERNAL_TRIGGER = "external_trigger"
    PREDICTIVE_WAKEUP = "predictive_wakeup"
    MAINTENANCE = "maintenance"


@dataclass
class SchedulingDecision:
    """Decision made by the scheduler"""
    should_wakeup: bool
    sleep_duration_minutes: float
    wakeup_reason: WakeupReason
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerMetrics:
    """Performance metrics for the scheduler"""
    total_wakeups: int = 0
    scheduled_wakeups: int = 0
    threshold_wakeups: int = 0
    activity_based_wakeups: int = 0
    battery_conservative_actions: int = 0
    predictive_wakeups: int = 0
    average_sleep_duration: float = 0.0
    energy_efficiency_score: float = 1.0
    response_time_avg: float = 0.0
    missed_events: int = 0


class Scheduler(ABC):
    """
    Abstract base class for sleep scheduling algorithms

    Features:
    - Pluggable scheduling strategies
    - Metrics collection
    - Decision logging
    - Configurable parameters
    """

    def __init__(self, name: str):
        """
        Initialize scheduler

        Args:
            name: Scheduler name for identification
        """
        self.name = name
        self.last_wakeup_time = 0.0
        self.last_sleep_time = 0.0
        self.metrics = SchedulerMetrics()
        self.decision_history: List[SchedulingDecision] = []
        self.is_active = True

    @abstractmethod
    def should_wakeup(self, current_time: float, node_state: Dict[str, Any]) -> SchedulingDecision:
        """
        Determine if node should wake up

        Args:
            current_time: Current simulation time
            node_state: Current node state including battery, sensors, etc.

        Returns:
            SchedulingDecision with wake-up recommendation
        """
        pass

    @abstractmethod
    def get_sleep_duration(self, current_time: float, node_state: Dict[str, Any]) -> float:
        """
        Calculate sleep duration after active period

        Args:
            current_time: Current simulation time
            node_state: Current node state

        Returns:
            Sleep duration in minutes
        """
        pass

    def record_wakeup(self, current_time: float, reason: WakeupReason):
        """
        Record a wake-up event for metrics

        Args:
            current_time: Wake-up time
            reason: Reason for wake-up
        """
        self.last_wakeup_time = current_time
        self.metrics.total_wakeups += 1

        if reason == WakeupReason.SCHEDULED_TIMER:
            self.metrics.scheduled_wakeups += 1
        elif reason == WakeupReason.THRESHOLD_BREACH:
            self.metrics.threshold_wakeups += 1
        elif reason == WakeupReason.HIGH_ACTIVITY:
            self.metrics.activity_based_wakeups += 1
        elif reason in [WakeupReason.BATTERY_CONSERVATIVE, WakeupReason.BATTERY_CRITICAL]:
            self.metrics.battery_conservative_actions += 1
        elif reason == WakeupReason.PREDICTIVE_WAKEUP:
            self.metrics.predictive_wakeups += 1

    def reset_metrics(self):
        """Reset scheduler metrics"""
        self.metrics = SchedulerMetrics()
        self.decision_history.clear()

    def get_efficiency_score(self) -> float:
        """
        Calculate efficiency score based on metrics

        Returns:
            Efficiency score (higher is better)
        """
        if self.metrics.total_wakeups == 0:
            return 1.0

        # Calculate optimal wake-up ratio
        optimal_ratio = 0.3  # 30% of wake-ups should be threshold-based
        threshold_ratio = self.metrics.threshold_wakeups / self.metrics.total_wakeups
        ratio_score = 1.0 - abs(threshold_ratio - optimal_ratio)

        # Calculate sleep efficiency
        sleep_efficiency = min(1.0, self.metrics.average_sleep_duration / 10.0)  # Optimal: 10 minutes

        # Battery conservation bonus
        battery_bonus = 1.0 + (self.metrics.battery_conservative_actions / max(1, self.metrics.total_wakeups)) * 0.5

        return ratio_score * sleep_efficiency * battery_bonus

    def get_summary(self) -> Dict[str, Any]:
        """
        Get scheduler performance summary

        Returns:
            Dictionary containing scheduler summary
        """
        return {
            'name': self.name,
            'is_active': self.is_active,
            'total_wakeups': self.metrics.total_wakeups,
            'wakeup_breakdown': {
                'scheduled': self.metrics.scheduled_wakeups,
                'threshold': self.metrics.threshold_wakeups,
                'activity_based': self.metrics.activity_based_wakeups,
                'battery_conservative': self.metrics.battery_conservative_actions,
                'predictive': self.metrics.predictive_wakeups
            },
            'average_sleep_duration': self.metrics.average_sleep_duration,
            'efficiency_score': self.get_efficiency_score(),
            'total_decisions': len(self.decision_history),
            'last_wakeup_time': self.last_wakeup_time
        }


class StaticDutyCycleScheduler(Scheduler):
    """
    Static duty cycle scheduler with fixed wake-up intervals

    Features:
    - Fixed sleep/wake cycles
    - Timer-based wake-ups only
    - Simple and predictable behavior
    - Configurable duty cycle
    """

    def __init__(self, sleep_interval_minutes: float = None):
        """
        Initialize static duty cycle scheduler

        Args:
            sleep_interval_minutes: Fixed sleep interval between wake-ups
        """
        super().__init__("StaticDutyCycle")
        self.sleep_interval_minutes = sleep_interval_minutes or config.scheduling.STATIC_DUTY_CYCLE_MINUTES
        self.sleep_interval_seconds = config.get_seconds_from_minutes(self.sleep_interval_minutes)

    def should_wakeup(self, current_time: float, node_state: Dict[str, Any]) -> SchedulingDecision:
        """
        Check if node should wake up based on fixed timer

        Args:
            current_time: Current simulation time
            node_state: Current node state (not used in static scheduler)

        Returns:
            SchedulingDecision for timer-based wake-up
        """
        if not self.is_active:
            return SchedulingDecision(
                should_wakeup=False,
                sleep_duration_minutes=self.sleep_interval_minutes,
                wakeup_reason=WakeupReason.SCHEDULED_TIMER,
                confidence=1.0
            )

        time_since_wakeup = current_time - self.last_wakeup_time

        if time_since_wakeup >= self.sleep_interval_seconds:
            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=0,
                wakeup_reason=WakeupReason.SCHEDULED_TIMER,
                confidence=1.0,
                metadata={'time_since_wakeup': time_since_wakeup}
            )

        return SchedulingDecision(
            should_wakeup=False,
            sleep_duration_minutes=self.sleep_interval_minutes,
            wakeup_reason=WakeupReason.SCHEDULED_TIMER,
            confidence=1.0
        )

    def get_sleep_duration(self, current_time: float, node_state: Dict[str, Any]) -> float:
        """
        Get fixed sleep duration

        Args:
            current_time: Current simulation time
            node_state: Current node state (not used in static scheduler)

        Returns:
            Fixed sleep duration in minutes
        """
        return self.sleep_interval_minutes


class AdaptiveDutyCycleScheduler(Scheduler):
    """
    Adaptive duty cycle scheduler with intelligent wake-up triggers

    Features:
    - Battery-aware scheduling
    - Threshold-based wake-ups
    - Activity pattern learning
    - Predictive wake-ups
    - Environmental adaptation
    - Multi-factor decision making
    """

    def __init__(self, enable_prediction: bool = True):
        """
        Initialize adaptive duty cycle scheduler

        Args:
            enable_prediction: Enable predictive wake-up capabilities
        """
        super().__init__("AdaptiveDutyCycle")
        self.enable_prediction = enable_prediction

        # Adaptive parameters
        self.current_sleep_duration = config.scheduling.ADAPTIVE_DEFAULT_SLEEP_MINUTES
        self.min_sleep_duration = config.scheduling.ADAPTIVE_MIN_SLEEP_MINUTES
        self.max_sleep_duration = config.scheduling.ADAPTIVE_MAX_SLEEP_MINUTES

        # Trend analysis
        self.sensor_trends: Dict[str, List[float]] = {}
        self.activity_patterns: List[Tuple[float, float]] = []  # (time, activity_level)
        self.prediction_window = config.scheduling.TREND_ANALYSIS_WINDOW

        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_factor = 1.0
        self.threshold_sensitivity = 1.0

    def should_wakeup(self, current_time: float, node_state: Dict[str, Any]) -> SchedulingDecision:
        """
        Determine if node should wake up using adaptive algorithm

        Args:
            current_time: Current simulation time
            node_state: Current node state with battery, sensors, etc.

        Returns:
            SchedulingDecision with adaptive reasoning
        """
        if not self.is_active:
            return SchedulingDecision(
                should_wakeup=False,
                sleep_duration_minutes=self.max_sleep_duration,
                wakeup_reason=WakeupReason.BATTERY_CONSERVATIVE,
                confidence=1.0
            )

        # Extract relevant information from node state
        battery_percentage = node_state.get('battery_percentage', 100.0)
        sensor_data = node_state.get('sensor_data', {})
        sensor_changes = node_state.get('sensor_changes', {})
        activity_level = node_state.get('activity_level', 0.0)

        # Priority 1: Critical battery conditions
        if battery_percentage <= config.scheduling.BATTERY_CRITICAL_THRESHOLD:
            return SchedulingDecision(
                should_wakeup=False,
                sleep_duration_minutes=self.max_sleep_duration,
                wakeup_reason=WakeupReason.BATTERY_CRITICAL,
                confidence=1.0,
                metadata={'battery_percentage': battery_percentage}
            )

        # Priority 2: Environmental threshold breaches
        threshold_decision = self._check_thresholds(sensor_data, current_time)
        if threshold_decision.should_wakeup:
            return threshold_decision

        # Priority 3: High activity detection
        activity_decision = self._check_activity_level(activity_level, sensor_changes, current_time)
        if activity_decision.should_wakeup:
            return activity_decision

        # Priority 4: Predictive wake-up (if enabled)
        if self.enable_prediction:
            prediction_decision = self._predictive_wakeup(sensor_data, current_time)
            if prediction_decision.should_wakeup:
                return prediction_decision

        # Priority 5: Battery-aware scheduling
        battery_decision = self._battery_aware_scheduling(battery_percentage, current_time)
        if battery_decision.should_wakeup:
            return battery_decision

        # Priority 6: Timer-based wake-up with adaptive interval
        time_since_wakeup = current_time - self.last_wakeup_time
        required_sleep_seconds = config.get_seconds_from_minutes(self.current_sleep_duration)

        if time_since_wakeup >= required_sleep_seconds:
            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=0,
                wakeup_reason=WakeupReason.SCHEDULED_TIMER,
                confidence=0.7,
                metadata={
                    'current_sleep_duration': self.current_sleep_duration,
                    'time_since_wakeup': time_since_wakeup
                }
            )

        return SchedulingDecision(
            should_wakeup=False,
            sleep_duration_minutes=self.current_sleep_duration,
            wakeup_reason=WakeupReason.SCHEDULED_TIMER,
            confidence=0.7
        )

    def get_sleep_duration(self, current_time: float, node_state: Dict[str, Any]) -> float:
        """
        Calculate adaptive sleep duration based on current conditions

        Args:
            current_time: Current simulation time
            node_state: Current node state

        Returns:
            Adaptive sleep duration in minutes
        """
        battery_percentage = node_state.get('battery_percentage', 100.0)
        sensor_changes = node_state.get('sensor_changes', {})
        activity_level = node_state.get('activity_level', 0.0)

        # Base sleep duration
        base_duration = self.current_sleep_duration

        # Battery-based adjustments
        if battery_percentage < config.scheduling.BATTERY_LOW_THRESHOLD:
            battery_multiplier = config.scheduling.BATTERY_CONSERVATION_MULTIPLIER
        elif battery_percentage < 50.0:
            battery_multiplier = 1.5
        else:
            battery_multiplier = 1.0

        # Activity-based adjustments
        if activity_level > config.scheduling.HIGH_CHANGE_RATE_THRESHOLD:
            activity_multiplier = 0.5  # Reduce sleep for high activity
        elif activity_level > 0.2:
            activity_multiplier = 0.8  # Moderate reduction
        else:
            activity_multiplier = 1.0

        # Time-based adjustments
        hours = (current_time / 3600) % 24
        if 6 <= hours <= 18:  # Daytime
            time_multiplier = config.scheduling.DAY_ACTIVITY_BOOST
        else:  # Nighttime
            time_multiplier = config.scheduling.NIGHT_SLEEP_EXTENSION

        # Calculate final sleep duration
        final_duration = base_duration * battery_multiplier * activity_multiplier * time_multiplier

        # Apply constraints
        final_duration = max(self.min_sleep_duration, min(self.max_sleep_duration, final_duration))

        # Update current sleep duration with learning
        self._update_sleep_duration_learning(final_duration)

        return final_duration

    def _check_thresholds(self, sensor_data: Dict[str, float], current_time: float) -> SchedulingDecision:
        """Check for environmental threshold breaches"""
        temperature = sensor_data.get('Temperature', 25.0)
        humidity = sensor_data.get('Humidity', 50.0)
        light = sensor_data.get('Light', 100.0)

        # Temperature thresholds
        if temperature > config.scheduling.THRESHOLD_TEMP_HIGH:
            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=0,
                wakeup_reason=WakeupReason.THRESHOLD_BREACH,
                confidence=1.0,
                metadata={'threshold_type': 'temperature_high', 'value': temperature}
            )
        elif temperature < config.scheduling.THRESHOLD_TEMP_LOW:
            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=0,
                wakeup_reason=WakeupReason.THRESHOLD_BREACH,
                confidence=1.0,
                metadata={'threshold_type': 'temperature_low', 'value': temperature}
            )

        # Humidity thresholds
        if humidity > config.scheduling.THRESHOLD_HUMIDITY_HIGH:
            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=0,
                wakeup_reason=WakeupReason.THRESHOLD_BREACH,
                confidence=1.0,
                metadata={'threshold_type': 'humidity_high', 'value': humidity}
            )
        elif humidity < config.scheduling.THRESHOLD_HUMIDITY_LOW:
            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=0,
                wakeup_reason=WakeupReason.THRESHOLD_BREACH,
                confidence=1.0,
                metadata={'threshold_type': 'humidity_low', 'value': humidity}
            )

        # Light thresholds
        if light > config.scheduling.THRESHOLD_LIGHT_HIGH:
            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=0,
                wakeup_reason=WakeupReason.THRESHOLD_BREACH,
                confidence=1.0,
                metadata={'threshold_type': 'light_high', 'value': light}
            )
        elif light < config.scheduling.THRESHOLD_LIGHT_LOW:
            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=0,
                wakeup_reason=WakeupReason.THRESHOLD_BREACH,
                confidence=1.0,
                metadata={'threshold_type': 'light_low', 'value': light}
            )

        return SchedulingDecision(
            should_wakeup=False,
            sleep_duration_minutes=self.current_sleep_duration,
            wakeup_reason=WakeupReason.SCHEDULED_TIMER,
            confidence=1.0
        )

    def _check_activity_level(self, activity_level: float, sensor_changes: Dict[str, bool],
                            current_time: float) -> SchedulingDecision:
        """Check for high activity levels that require immediate attention"""
        # Check for significant sensor changes
        has_significant_changes = any(sensor_changes.values())

        if activity_level > config.scheduling.HIGH_CHANGE_RATE_THRESHOLD or has_significant_changes:
            # Shorten sleep duration for high activity
            reduced_sleep = max(self.min_sleep_duration, self.current_sleep_duration * 0.5)

            return SchedulingDecision(
                should_wakeup=True,
                sleep_duration_minutes=reduced_sleep,
                wakeup_reason=WakeupReason.HIGH_ACTIVITY,
                confidence=0.9,
                metadata={
                    'activity_level': activity_level,
                    'sensor_changes': sensor_changes,
                    'reduced_sleep': reduced_sleep
                }
            )

        return SchedulingDecision(
            should_wakeup=False,
            sleep_duration_minutes=self.current_sleep_duration,
            wakeup_reason=WakeupReason.SCHEDULED_TIMER,
            confidence=1.0
        )

    def _predictive_wakeup(self, sensor_data: Dict[str, float], current_time: float) -> SchedulingDecision:
        """Predictive wake-up based on trend analysis"""
        # Update sensor trends
        for sensor_name, value in sensor_data.items():
            if sensor_name not in self.sensor_trends:
                self.sensor_trends[sensor_name] = []
            self.sensor_trends[sensor_name].append(value)

            # Keep only recent values
            if len(self.sensor_trends[sensor_name]) > self.prediction_window * 2:
                self.sensor_trends[sensor_name] = self.sensor_trends[sensor_name][-self.prediction_window * 2:]

        # Analyze trends for each sensor
        for sensor_name, values in self.sensor_trends.items():
            if len(values) < self.prediction_window:
                continue

            recent_values = values[-self.prediction_window:]
            trend = self._detect_trend(recent_values)

            # Predictive wake-up for rapidly changing values
            if trend in ['increasing_rapid', 'decreasing_rapid']:
                predicted_time_to_threshold = self._predict_threshold_crossing(sensor_name, recent_values)

                if predicted_time_to_threshold < self.current_sleep_duration:
                    return SchedulingDecision(
                        should_wakeup=True,
                        sleep_duration_minutes=predicted_time_to_threshold * 0.8,  # Wake up early
                        wakeup_reason=WakeupReason.PREDICTIVE_WAKEUP,
                        confidence=0.7,
                        metadata={
                            'sensor': sensor_name,
                            'trend': trend,
                            'predicted_time': predicted_time_to_threshold
                        }
                    )

        return SchedulingDecision(
            should_wakeup=False,
            sleep_duration_minutes=self.current_sleep_duration,
            wakeup_reason=WakeupReason.SCHEDULED_TIMER,
            confidence=1.0
        )

    def _battery_aware_scheduling(self, battery_percentage: float, current_time: float) -> SchedulingDecision:
        """Battery-aware scheduling with conservative operation"""
        if battery_percentage < config.scheduling.BATTERY_LOW_THRESHOLD:
            # Extend sleep duration for battery conservation
            extended_sleep = min(self.max_sleep_duration, self.current_sleep_duration * 2.0)

            return SchedulingDecision(
                should_wakeup=False,
                sleep_duration_minutes=extended_sleep,
                wakeup_reason=WakeupReason.BATTERY_CONSERVATIVE,
                confidence=1.0,
                metadata={
                    'battery_percentage': battery_percentage,
                    'extended_sleep': extended_sleep
                }
            )

        return SchedulingDecision(
            should_wakeup=False,
            sleep_duration_minutes=self.current_sleep_duration,
            wakeup_reason=WakeupReason.SCHEDULED_TIMER,
            confidence=1.0
        )

    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend in sensor values"""
        if len(values) < 3:
            return 'insufficient_data'

        # Simple linear regression
        x = list(range(len(values)))
        y = values
        n = len(values)

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        if n * sum_x2 - sum_x ** 2 == 0:
            return 'stable'

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        threshold = config.scheduling.TREND_THRESHOLD

        if slope > threshold * 2:
            return 'increasing_rapid'
        elif slope > threshold:
            return 'increasing'
        elif slope < -threshold * 2:
            return 'decreasing_rapid'
        elif slope < -threshold:
            return 'decreasing'
        else:
            return 'stable'

    def _predict_threshold_crossing(self, sensor_name: str, values: List[float]) -> float:
        """Predict time until threshold crossing"""
        # Get threshold for sensor
        thresholds = {
            'Temperature': (config.scheduling.THRESHOLD_TEMP_HIGH, config.scheduling.THRESHOLD_TEMP_LOW),
            'Humidity': (config.scheduling.THRESHOLD_HUMIDITY_HIGH, config.scheduling.THRESHOLD_HUMIDITY_LOW),
            'Light': (config.scheduling.THRESHOLD_LIGHT_HIGH, config.scheduling.THRESHOLD_LIGHT_LOW)
        }

        if sensor_name not in thresholds:
            return float('inf')

        high_threshold, low_threshold = thresholds[sensor_name]
        current_value = values[-1]

        # Calculate rate of change
        if len(values) < 2:
            return float('inf')

        rate = (values[-1] - values[-2])  # Change per reading

        if rate == 0:
            return float('inf')

        # Predict time to reach threshold
        if rate > 0:  # Increasing
            if current_value >= high_threshold:
                return 0
            time_to_threshold = (high_threshold - current_value) / rate
        else:  # Decreasing
            if current_value <= low_threshold:
                return 0
            time_to_threshold = (current_value - low_threshold) / abs(rate)

        return max(0, time_to_threshold)

    def _update_sleep_duration_learning(self, actual_sleep_duration: float):
        """Update sleep duration based on learning algorithm"""
        # Exponential moving average with learning rate
        self.current_sleep_duration = (1 - self.learning_rate) * self.current_sleep_duration + \
                                     self.learning_rate * actual_sleep_duration

        # Apply adaptation factor for gradual changes
        self.current_sleep_duration *= self.adaptation_factor

        # Ensure within bounds
        self.current_sleep_duration = max(self.min_sleep_duration,
                                         min(self.max_sleep_duration, self.current_sleep_duration))


class SchedulerManager:
    """Manager for multiple schedulers with performance comparison"""

    def __init__(self):
        self.schedulers: Dict[str, Scheduler] = {}
        self.active_scheduler_name: str = None

    def add_scheduler(self, scheduler: Scheduler, set_active: bool = False):
        """Add a scheduler to the manager"""
        self.schedulers[scheduler.name] = scheduler
        if set_active or not self.active_scheduler_name:
            self.active_scheduler_name = scheduler.name

    def set_active_scheduler(self, scheduler_name: str):
        """Set the active scheduler"""
        if scheduler_name in self.schedulers:
            self.active_scheduler_name = scheduler_name
            # Deactivate others
            for name, scheduler in self.schedulers.items():
                scheduler.is_active = (name == scheduler_name)

    def get_active_scheduler(self) -> Optional[Scheduler]:
        """Get the currently active scheduler"""
        if self.active_scheduler_name:
            return self.schedulers.get(self.active_scheduler_name)
        return None

    def get_scheduler_comparison(self) -> Dict[str, Dict]:
        """Get performance comparison of all schedulers"""
        comparison = {}
        for name, scheduler in self.schedulers.items():
            comparison[name] = {
                'summary': scheduler.get_summary(),
                'efficiency_score': scheduler.get_efficiency_score(),
                'is_active': scheduler.is_active
            }
        return comparison

    def reset_all_metrics(self):
        """Reset metrics for all schedulers"""
        for scheduler in self.schedulers.values():
            scheduler.reset_metrics()