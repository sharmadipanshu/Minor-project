"""
Logging System for IoT Sleep Scheduling Simulation
CSV-based data logging with performance optimization and comprehensive data collection
"""

import csv
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, TextIO
from dataclasses import dataclass, field
from pathlib import Path

from ..config import config


@dataclass
class LogEntry:
    """Single log entry with all required fields"""
    timestamp: float
    node_id: str
    mode: str
    energy_consumed_mah: float
    battery_percentage: float
    temperature: float
    humidity: float
    light_level: float
    wakeup_reason: str
    transmission_successful: bool
    voltage: float = 0.0
    current_draw_ma: float = 0.0
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingMetrics:
    """Metrics for logging system performance"""
    total_entries: int = 0
    entries_per_second: float = 0.0
    file_size_bytes: int = 0
    last_write_time: float = 0.0
    write_errors: int = 0
    buffer_size: int = 0


class SimulationLogger:
    """
    Comprehensive logging system for IoT sleep scheduling simulation

    Features:
    - CSV data logging with configurable intervals
    - Buffer-based writing for performance optimization
    - Multiple file formats (CSV, JSON)
    - Data validation and error handling
    - Real-time metrics collection
    - Rotating log files
    - Compressed archive support
    """

    def __init__(self,
                 log_file_path: str = None,
                 buffer_size: int = 100,
                 enable_json_backup: bool = True,
                 auto_flush_interval: float = 60.0):
        """
        Initialize simulation logger

        Args:
            log_file_path: Path to primary log file
            buffer_size: Number of entries to buffer before writing
            enable_json_backup: Create JSON backup of all data
            auto_flush_interval: Auto-flush buffer interval in seconds
        """
        # File paths
        self.base_path = Path(config.simulation.LOGS_OUTPUT_DIR)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Use default path if not specified
        if log_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = self.base_path / f"simulation_log_{timestamp}.csv"

        self.log_file_path = Path(log_file_path)
        self.json_backup_path = self.log_file_path.with_suffix('.json')

        # Configuration
        self.buffer_size = buffer_size
        self.enable_json_backup = enable_json_backup
        self.auto_flush_interval = auto_flush_interval

        # State
        self.csv_file: Optional[TextIO] = None
        self.csv_writer: Optional[csv.DictWriter] = None
        self.is_initialized = False
        self.buffer: List[LogEntry] = []
        self.last_flush_time = 0.0

        # Metrics
        self.metrics = LoggingMetrics()

        # Headers for CSV file
        self.csv_headers = [
            'timestamp',
            'node_id',
            'mode',
            'energy_consumed_mah',
            'battery_percentage',
            'temperature',
            'humidity',
            'light_level',
            'wakeup_reason',
            'transmission_successful',
            'voltage',
            'current_draw_ma'
        ]

        # JSON data storage
        self.json_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'simulation_config': config.to_dict(),
                'version': '1.0'
            },
            'entries': []
        }

    def initialize(self) -> bool:
        """
        Initialize logging system and create files

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create CSV file
            self.csv_file = open(self.log_file_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_headers)
            self.csv_writer.writeheader()

            # Store initial file size
            self.metrics.file_size_bytes = self.log_file_path.stat().st_size

            self.is_initialized = True
            print(f"Simulation logger initialized: {self.log_file_path}")

            return True

        except Exception as e:
            print(f"Failed to initialize logger: {e}")
            self.metrics.write_errors += 1
            return False

    def log_entry(self, data: Dict[str, Any]) -> bool:
        """
        Log a single data entry

        Args:
            data: Dictionary containing log data

        Returns:
            True if logging successful, False otherwise
        """
        if not self.is_initialized:
            if not self.initialize():
                return False

        try:
            # Create log entry with validation
            entry = self._create_log_entry(data)

            # Add to buffer
            self.buffer.append(entry)
            self.metrics.buffer_size = len(self.buffer)

            # Check if buffer should be flushed
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer()

            # Check auto-flush interval
            current_time = data.get('timestamp', 0.0)
            if current_time - self.last_flush_time >= self.auto_flush_interval:
                self.flush_buffer()

            # Add to JSON data
            if self.enable_json_backup:
                self.json_data['entries'].append({
                    'timestamp': entry.timestamp,
                    'node_id': entry.node_id,
                    'mode': entry.mode,
                    'data': {
                        'energy_consumed_mah': entry.energy_consumed_mah,
                        'battery_percentage': entry.battery_percentage,
                        'temperature': entry.temperature,
                        'humidity': entry.humidity,
                        'light_level': entry.light_level,
                        'wakeup_reason': entry.wakeup_reason,
                        'transmission_successful': entry.transmission_successful,
                        'voltage': entry.voltage,
                        'current_draw_ma': entry.current_draw_ma,
                        'additional_data': entry.additional_data
                    }
                })

            self.metrics.total_entries += 1
            return True

        except Exception as e:
            print(f"Failed to log entry: {e}")
            self.metrics.write_errors += 1
            return False

    def log_batch(self, entries: List[Dict[str, Any]]) -> bool:
        """
        Log multiple entries efficiently

        Args:
            entries: List of data dictionaries

        Returns:
            True if all entries logged successfully, False otherwise
        """
        success_count = 0
        for entry in entries:
            if self.log_entry(entry):
                success_count += 1

        return success_count == len(entries)

    def flush_buffer(self) -> bool:
        """
        Flush buffered entries to file

        Returns:
            True if flush successful, False otherwise
        """
        if not self.buffer or not self.csv_writer:
            return True

        try:
            # Convert entries to CSV rows
            rows = []
            for entry in self.buffer:
                row = {
                    'timestamp': entry.timestamp,
                    'node_id': entry.node_id,
                    'mode': entry.mode,
                    'energy_consumed_mah': entry.energy_consumed_mah,
                    'battery_percentage': entry.battery_percentage,
                    'temperature': entry.temperature,
                    'humidity': entry.humidity,
                    'light_level': entry.light_level,
                    'wakeup_reason': entry.wakeup_reason,
                    'transmission_successful': entry.transmission_successful,
                    'voltage': entry.voltage,
                    'current_draw_ma': entry.current_draw_ma
                }
                rows.append(row)

            # Write all rows
            self.csv_writer.writerows(rows)
            self.csv_file.flush()

            # Update metrics
            self.metrics.file_size_bytes = self.log_file_path.stat().st_size
            self.last_flush_time = self.buffer[-1].timestamp if self.buffer else 0.0
            self.buffer.clear()
            self.metrics.buffer_size = 0

            return True

        except Exception as e:
            print(f"Failed to flush buffer: {e}")
            self.metrics.write_errors += 1
            return False

    def close(self) -> bool:
        """
        Close logger and finalize files

        Returns:
            True if closure successful, False otherwise
        """
        try:
            # Flush any remaining buffer
            if self.buffer:
                self.flush_buffer()

            # Write JSON backup
            if self.enable_json_backup and self.json_data['entries']:
                self._write_json_backup()

            # Close CSV file
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None

            print(f"Logger closed. {self.metrics.total_entries} entries written.")
            return True

        except Exception as e:
            print(f"Failed to close logger: {e}")
            self.metrics.write_errors += 1
            return False

    def _create_log_entry(self, data: Dict[str, Any]) -> LogEntry:
        """Create validated log entry from data dictionary"""
        # Extract and validate required fields
        timestamp = float(data.get('timestamp', 0.0))
        node_id = str(data.get('node_id', 'unknown'))
        mode = str(data.get('mode', 'unknown'))
        energy_consumed_mah = float(data.get('energy_consumed_mah', 0.0))
        battery_percentage = float(data.get('battery_percentage', 0.0))
        temperature = float(data.get('temperature', 0.0))
        humidity = float(data.get('humidity', 0.0))
        light_level = float(data.get('light_level', 0.0))
        wakeup_reason = str(data.get('wakeup_reason', 'unknown'))
        transmission_successful = bool(data.get('transmission_successful', False))

        # Optional fields
        voltage = float(data.get('voltage', 0.0))
        current_draw_ma = float(data.get('current_draw_ma', 0.0))

        # Extract additional data (fields not in standard header)
        additional_data = {}
        reserved_fields = {
            'timestamp', 'node_id', 'mode', 'energy_consumed_mah', 'battery_percentage',
            'temperature', 'humidity', 'light_level', 'wakeup_reason', 'transmission_successful',
            'voltage', 'current_draw_ma'
        }

        for key, value in data.items():
            if key not in reserved_fields:
                additional_data[key] = value

        return LogEntry(
            timestamp=timestamp,
            node_id=node_id,
            mode=mode,
            energy_consumed_mah=energy_consumed_mah,
            battery_percentage=battery_percentage,
            temperature=temperature,
            humidity=humidity,
            light_level=light_level,
            wakeup_reason=wakeup_reason,
            transmission_successful=transmission_successful,
            voltage=voltage,
            current_draw_ma=current_draw_ma,
            additional_data=additional_data
        )

    def _write_json_backup(self) -> bool:
        """Write JSON backup of all logged data"""
        try:
            # Update metadata
            self.json_data['metadata']['entries_count'] = len(self.json_data['entries'])
            self.json_data['metadata']['completed_at'] = datetime.now().isoformat()

            # Write JSON file
            with open(self.json_backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Failed to write JSON backup: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logging system statistics

        Returns:
            Dictionary containing logging statistics
        """
        # Calculate entries per second
        if self.metrics.total_entries > 0 and self.metrics.last_flush_time > 0:
            self.metrics.entries_per_second = self.metrics.total_entries / self.metrics.last_flush_time

        return {
            'total_entries': self.metrics.total_entries,
            'entries_per_second': self.metrics.entries_per_second,
            'file_size_bytes': self.metrics.file_size_bytes,
            'file_size_mb': self.metrics.file_size_bytes / (1024 * 1024),
            'buffer_size': self.metrics.buffer_size,
            'write_errors': self.metrics.write_errors,
            'is_initialized': self.is_initialized,
            'log_file_path': str(self.log_file_path),
            'json_backup_enabled': self.enable_json_backup,
            'json_backup_path': str(self.json_backup_path) if self.enable_json_backup else None
        }

    def export_to_different_format(self, output_format: str, output_path: str = None) -> bool:
        """
        Export logged data to different format

        Args:
            output_format: Target format ('csv', 'json', 'excel', 'parquet')
            output_path: Output file path (auto-generated if not provided)

        Returns:
            True if export successful, False otherwise
        """
        try:
            if output_format.lower() == 'json':
                return self._export_json(output_path)
            elif output_format.lower() == 'csv':
                return self._export_csv(output_path)
            elif output_format.lower() == 'excel':
                return self._export_excel(output_path)
            else:
                print(f"Unsupported export format: {output_format}")
                return False

        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def _export_json(self, output_path: str = None) -> bool:
        """Export to JSON format"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.base_path / f"simulation_export_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)

        print(f"JSON export completed: {output_path}")
        return True

    def _export_csv(self, output_path: str = None) -> bool:
        """Export to CSV format (different structure)"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.base_path / f"simulation_export_{timestamp}.csv"

        # Create enhanced CSV with all data
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if self.json_data['entries']:
                # Get all possible fields
                all_fields = set(self.csv_headers)
                for entry in self.json_data['entries']:
                    all_fields.update(entry['data'].get('additional_data', {}).keys())

                writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
                writer.writeheader()

                for entry in self.json_data['entries']:
                    row = {
                        'timestamp': entry['timestamp'],
                        'node_id': entry['node_id'],
                        'mode': entry['mode']
                    }
                    row.update(entry['data'])
                    writer.writerow(row)

        print(f"CSV export completed: {output_path}")
        return True

    def _export_excel(self, output_path: str = None) -> bool:
        """Export to Excel format (if pandas available)"""
        try:
            import pandas as pd
        except ImportError:
            print("pandas not available for Excel export")
            return False

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.base_path / f"simulation_export_{timestamp}.xlsx"

        # Convert to DataFrame
        if self.json_data['entries']:
            data = []
            for entry in self.json_data['entries']:
                row = {
                    'timestamp': entry['timestamp'],
                    'node_id': entry['node_id'],
                    'mode': entry['mode']
                }
                row.update(entry['data'])
                data.append(row)

            df = pd.DataFrame(data)
            df.to_excel(output_path, index=True)
            print(f"Excel export completed: {output_path}")
            return True

        return False

    def analyze_log_data(self) -> Dict[str, Any]:
        """
        Perform basic analysis on logged data

        Returns:
            Dictionary containing analysis results
        """
        if not self.json_data['entries']:
            return {'error': 'No data to analyze'}

        # Extract data for analysis
        timestamps = [entry['timestamp'] for entry in self.json_data['entries']]
        battery_levels = [entry['data']['battery_percentage'] for entry in self.json_data['entries']]
        energy_consumed = [entry['data']['energy_consumed_mah'] for entry in self.json_data['entries']]
        temperatures = [entry['data']['temperature'] for entry in self.json_data['entries']]

        # Basic statistics
        analysis = {
            'total_entries': len(self.json_data['entries']),
            'time_span_seconds': max(timestamps) - min(timestamps) if timestamps else 0,
            'battery_stats': {
                'average': sum(battery_levels) / len(battery_levels) if battery_levels else 0,
                'min': min(battery_levels) if battery_levels else 0,
                'max': max(battery_levels) if battery_levels else 0
            },
            'energy_stats': {
                'total_consumed': sum(energy_consumed),
                'average_per_entry': sum(energy_consumed) / len(energy_consumed) if energy_consumed else 0
            },
            'temperature_stats': {
                'average': sum(temperatures) / len(temperatures) if temperatures else 0,
                'min': min(temperatures) if temperatures else 0,
                'max': max(temperatures) if temperatures else 0
            }
        }

        # Node-wise analysis
        node_data = {}
        for entry in self.json_data['entries']:
            node_id = entry['node_id']
            if node_id not in node_data:
                node_data[node_id] = {
                    'entries': 0,
                    'total_energy': 0.0,
                    'min_battery': 100.0,
                    'max_battery': 0.0
                }

            node_data[node_id]['entries'] += 1
            node_data[node_id]['total_energy'] += entry['data']['energy_consumed_mah']
            node_data[node_id]['min_battery'] = min(node_data[node_id]['min_battery'], entry['data']['battery_percentage'])
            node_data[node_id]['max_battery'] = max(node_data[node_id]['max_battery'], entry['data']['battery_percentage'])

        analysis['node_analysis'] = node_data

        return analysis


# Convenience function for creating logger
def create_logger(log_file_path: str = None, **kwargs) -> SimulationLogger:
    """
    Create and initialize a simulation logger

    Args:
        log_file_path: Path to log file
        **kwargs: Additional arguments for SimulationLogger

    Returns:
        Initialized SimulationLogger instance
    """
    logger = SimulationLogger(log_file_path=log_file_path, **kwargs)
    logger.initialize()
    return logger


# Context manager for automatic logger management
class LoggingContext:
    """Context manager for automatic logger setup and cleanup"""

    def __init__(self, log_file_path: str = None, **kwargs):
        self.logger = SimulationLogger(log_file_path=log_file_path, **kwargs)

    def __enter__(self) -> SimulationLogger:
        self.logger.initialize()
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()
        return False