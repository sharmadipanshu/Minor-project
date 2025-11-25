# System Architecture
## Energy-Efficient Sleep Scheduling and Power Optimization for IoT Sensor Nodes

---

## 1. Overview

The IoT Sleep Scheduling System is designed as a modular, extensible framework for simulating and optimizing energy consumption in wireless sensor networks. The architecture separates concerns into distinct layers while maintaining loose coupling between components.

### 1.1 Design Principles
- **Modularity**: Each component has a single responsibility
- **Extensibility**: New schedulers and sensors can be easily added
- **Testability**: All components can be unit tested independently
- **Performance**: Optimized for large-scale simulations
- **Maintainability**: Clear interfaces and well-documented code

### 1.2 Architecture Goals
- Support multiple scheduling algorithms
- Realistic energy modeling
- Scalable simulation engine
- Comprehensive data analysis
- Flexible configuration management

---

## 2. High-Level Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        CLI[Command Line Interface]
        API[Python API]
        GUI[Graphical Interface]
    end

    subgraph "Simulation Engine"
        SE[Simulator Engine]
        ENV[Environment Simulator]
        TIME[Time Manager]
    end

    subgraph "Core Components"
        NM[Node Manager]
        BM[Battery Manager]
        SM[Sensor Manager]
        PM[Power Manager]
    end

    subgraph "Algorithm Layer"
        SS[Scheduling System]
        SA[Static Scheduler]
        AA[Adaptive Scheduler]
    end

    subgraph "Data Layer"
        LOG[Logging System]
        OUT[Output Generation]
        ANA[Analysis Engine]
    end

    subgraph "Configuration"
        CFG[Configuration Manager]
        PARAM[Parameter Store]
    end

    CLI --> SE
    API --> SE
    SE --> NM
    SE --> ENV
    SE --> TIME
    NM --> BM
    NM --> SM
    NM --> PM
    NM --> SS
    SS --> SA
    SS --> AA
    SE --> LOG
    LOG --> OUT
    OUT --> ANA
    CFG --> PARAM
    SE --> CFG
```

---

## 3. Component Architecture

### 3.1 Simulation Engine

The simulation engine orchestrates all components and manages the simulation timeline.

```mermaid
classDiagram
    class Simulator {
        -current_time: float
        -nodes: Dict[str, IoTNode]
        -environment: EnvironmentSimulator
        -logger: SimulationLogger
        +add_node(node)
        +run(duration)
        +step(current_time)
        -process_events()
    }

    class EnvironmentSimulator {
        -base_temperature: float
        -weather_conditions: str
        -temperature_history: List
        +get_environment_state(time)
        +update_weather()
        +trigger_weather_event()
    }

    Simulator --> EnvironmentSimulator
```

**Key Responsibilities:**
- Time management and event scheduling
- Node lifecycle management
- Environmental condition simulation
- Data collection and logging
- Performance monitoring

### 3.2 Node Management System

The node management system handles all IoT sensor node operations and state transitions.

```mermaid
stateDiagram-v2
    [*] --> Initializing
    Initializing --> Active
    Active --> Sensing
    Sensing --> Processing
    Processing --> Transmitting
    Transmitting --> Active
    Active --> LightSleep
    LightSleep --> DeepSleep
    DeepSleep --> Active
    Active --> Shutdown
    LightSleep --> Shutdown
    DeepSleep --> Shutdown
    Shutdown --> [*]
```

**Core Classes:**

```mermaid
classDiagram
    class IoTNode {
        -node_id: str
        -battery: Battery
        -sensors: SensorManager
        -scheduler: Scheduler
        -current_state: NodeState
        +update(current_time, environment)
        +wakeup(reason)
        +sleep(duration)
        +sample_sensors()
        +transmit_data()
        +process_data()
    }

    class SensorManager {
        -sensors: Dict[str, Sensor]
        -active_sensors: List[str]
        +add_sensor(sensor)
        +read_all_sensors(time, environment)
        +detect_significant_changes()
        +get_overall_activity_level()
    }

    class PowerModel {
        -power_parameters: PowerConfig
        -consumption_history: List
        +calculate_consumption(state, duration)
        +get_current_draw(state)
        +estimate_lifetime(node, rate)
    }

    IoTNode --> SensorManager
    IoTNode --> PowerModel
```

### 3.3 Scheduling System

The scheduling system implements different duty cycling algorithms with pluggable architecture.

```mermaid
classDiagram
    class Scheduler {
        <<abstract>>
        +should_wakeup(current_time, node_state)
        +get_sleep_duration(current_time, node_state)
        +record_wakeup(current_time, reason)
    }

    class StaticDutyCycleScheduler {
        -sleep_interval: float
        -last_wakeup: float
        +should_wakeup(current_time, node_state)
        +get_sleep_duration(current_time, node_state)
    }

    class AdaptiveDutyCycleScheduler {
        -current_sleep_duration: float
        -sensor_trends: Dict
        -activity_patterns: List
        +should_wakeup(current_time, node_state)
        +get_sleep_duration(current_time, node_state)
        -check_thresholds(sensor_data, time)
        -detect_trend(values)
    }

    Scheduler <|-- StaticDutyCycleScheduler
    Scheduler <|-- AdaptiveDutyCycleScheduler
```

**Algorithm Flow:**

```mermaid
flowchart TD
    Start --> [Get Node State]
    [Get Node State] --> {Battery Critical?}
    {Battery Critical?} -->|Yes| [Enter Deep Sleep]
    {Battery Critical?} -->|No| {Thresholds Breached?}
    {Thresholds Breached?} -->|Yes| [Immediate Wakeup]
    {Thresholds Breached?} -->|No| {High Activity?}
    {High Activity?} -->|Yes| [Reduce Sleep Duration]
    {High Activity?} -->|No| {Battery Low?}
    {Battery Low?} -->|Yes| [Extend Sleep Duration]
    {Battery Low?} -->|No| [Normal Sleep]
    [Immediate Wakeup] --> [Process Data]
    [Reduce Sleep Duration] --> [Process Data]
    [Normal Sleep] --> [Enter Sleep Mode]
    [Extend Sleep Duration] --> [Enter Deep Sleep]
    [Enter Sleep Mode] --> [Wait for Timer]
    [Enter Deep Sleep] --> [Wait for Timer]
    [Wait for Timer] --> [Wake Up]
    [Wake Up] --> [Process Data]
    [Process Data] --> [Transmit Data]
    [Transmit Data] --> Start
    [Enter Deep Sleep] --> Start
```

### 3.4 Battery Management

The battery management system provides realistic energy modeling with multiple battery types and characteristics.

```mermaid
classDiagram
    class Battery {
        -capacity_mah: float
        -current_charge_mah: float
        -voltage: float
        -discharge_curve: Dict
        -charge_history: List[BatteryState]
        +consume_current(current_ma, duration_seconds)
        +get_remaining_percentage()
        +get_voltage()
        +update(current_time, ambient_temp)
        -interpolate_voltage(percentage)
        -apply_temperature_effects()
    }

    class BatteryMetrics {
        -total_energy_consumed_mah: float
        -average_current_ma: float
        -peak_current_ma: float
        -discharge_rate_mah_per_hour: float
        -estimated_remaining_hours: float
    }

    Battery --> BatteryMetrics
```

**Discharge Curve Model:**

```mermaid
graph LR
    A[100% - 4.2V] --> B[90% - 4.0V]
    B --> C[80% - 3.9V]
    C --> D[70% - 3.8V]
    D --> E[60% - 3.7V]
    E --> F[50% - 3.6V]
    F --> G[40% - 3.5V]
    G --> H[30% - 3.4V]
    H --> I[20% - 3.3V]
    I --> J[10% - 3.1V]
    J --> K[5% - 3.0V]
    K --> L[0% - 2.7V]
```

### 3.5 Sensor System

The sensor system provides realistic data generation with environmental correlation and failure simulation.

```mermaid
classDiagram
    class Sensor {
        <<abstract>>
        -name: str
        -accuracy: float
        -noise_std: float
        +read_value(current_time, environment)
        +add_noise(value)
        +get_previous_value()
    }

    class TemperatureSensor {
        -base_temperature: float
        -daily_amplitude: float
        -weather_offset: float
        +read_value(current_time, environment)
        -calculate_daily_cycle(hours)
        -add_weather_effects(value)
    }

    class HumiditySensor {
        -base_humidity: float
        -daily_amplitude: float
        +read_value(current_time, environment)
        -calculate_humidity_cycle(hours)
        -apply_temperature_correlation(value, temp)
    }

    class LightSensor {
        -max_daylight: float
        -night_level: float
        +read_value(current_time, environment)
        -calculate_light_level(hours, day_of_year)
        -apply_weather_effects(level)
    }

    Sensor <|-- TemperatureSensor
    Sensor <|-- HumiditySensor
    Sensor <|-- LightSensor
```

---

## 4. Data Flow Architecture

### 4.1 Simulation Data Flow

```mermaid
sequenceDiagram
    participant CLI
    participant SE as Simulator Engine
    participant NM as Node Manager
    participant SS as Scheduler
    participant BM as Battery Manager
    participant LOG as Logger

    CLI->>SE: Initialize Simulation
    SE->>SE: Create Environment
    SE->>NM: Add Nodes
    loop Simulation Loop
        SE->>NM: Update Node
        NM->>SS: Get Sleep Decision
        SS-->>NM: Sleep/Wake Command
        NM->>BM: Update Battery
        BM-->>NM: Battery State
        NM-->>SE: Node Update
        SE->>LOG: Log Event
    end
    SE->>CLI: Return Results
```

### 4.2 Scheduling Decision Flow

```mermaid
sequenceDiagram
    participant Node
    participant Scheduler
    participant Battery
    participant Sensors
    participant Environment

    Node->>Scheduler: Request Decision
    Scheduler->>Battery: Get Battery Level
    Battery-->>Scheduler: Battery Percentage
    Scheduler->>Sensors: Get Current Data
    Sensors->>Environment: Get Conditions
    Environment-->>Sensors: Environmental Data
    Sensors-->>Scheduler: Sensor Readings
    Scheduler->>Scheduler: Analyze Factors
    Scheduler-->>Node: Sleep Duration / Wake-up
    Node->>Node: Execute Action
```

---

## 5. Configuration Architecture

### 5.1 Configuration Hierarchy

```mermaid
graph TD
    A[Global Configuration] --> B[Power Configuration]
    A --> C[Battery Configuration]
    A --> D[Sensor Configuration]
    A --> E[Scheduling Configuration]
    A --> F[Simulation Configuration]

    B --> G[Active Current]
    B --> H[Sleep Current]
    B --> I[TX Power]

    C --> J[Capacity]
    C --> K[Discharge Curve]
    C --> L[Efficiency Factor]

    D --> M[Temperature Range]
    D --> N[Humidity Range]
    D --> O[Light Range]
    D --> P[Accuracy]

    E --> Q[Static Interval]
    E --> R[Adaptive Parameters]
    E --> S[Thresholds]
    E --> T[Learning Rate]

    F --> U[Duration]
    F --> V[Time Step]
    F --> W[Logging Level]
```

### 5.2 Parameter Management

```mermaid
classDiagram
    class ConfigManager {
        -config_hierarchy: Dict
        -parameter_cache: Dict
        +load_configuration(file_path)
        +get_parameter(key, default)
        +set_parameter(key, value)
        +validate_configuration()
        +save_configuration(file_path)
    }

    class PowerConfig {
        -ACTIVE_CURRENT_MA: float
        -SLEEP_CURRENT_MA: float
        -DEEP_SLEEP_CURRENT_MA: float
        -TX_CURRENT_MA: float
    }

    class BatteryConfig {
        -BATTERY_CAPACITY_MAH: float
        -INITIAL_CHARGE_PERCENTAGE: float
        -DISCHARGE_CURVE: Dict
        -TEMPERATURE_EFFECTS: Dict
    }

    class SensorConfig {
        -TEMP_RANGE: Tuple[float, float]
        -HUMIDITY_RANGE: Tuple[float, float]
        -LIGHT_RANGE: Tuple[float, float]
        -ACCURACY_SPECIFICATIONS: Dict
    }

    ConfigManager --> PowerConfig
    ConfigManager --> BatteryConfig
    ConfigManager --> SensorConfig
```

---

## 6. Plugin Architecture

### 6.1 Scheduler Plugin Interface

```python
class SchedulerPlugin:
    """Base interface for scheduler plugins"""

    def initialize(self, config: Dict) -> bool:
        """Initialize scheduler with configuration"""
        pass

    def should_wakeup(self, current_time: float, node_state: Dict) -> SchedulingDecision:
        """Determine if node should wake up"""
        pass

    def get_sleep_duration(self, current_time: float, node_state: Dict) -> float:
        """Calculate sleep duration"""
        pass

    def cleanup(self) -> None:
        """Clean up resources"""
        pass
```

### 6.2 Sensor Plugin Architecture

```mermaid
classDiagram
    class SensorPlugin {
        <<interface>>
        +initialize(config)
        +read_value(current_time, environment)
        +calibrate(reference, measured)
        +get_accuracy()
    }

    class CustomSensor {
        -custom_parameters: Dict
        -calibration_data: List
        +initialize(config)
        +read_value(current_time, environment)
        +apply_custom_processing(value)
    }

    SensorPlugin <|-- CustomSensor
```

---

## 7. Performance Architecture

### 7.1 Optimization Strategies

**Memory Management:**
- Circular buffers for historical data
- Object pooling for frequently created objects
- Lazy loading of optional components
- Configurable data retention policies

**Execution Optimization:**
- Event-driven simulation architecture
- Parallelizable node updates
- Efficient time-stepping algorithm
- Cache frequently accessed calculations

**I/O Optimization:**
- Buffered logging with configurable flush intervals
- Batch processing of sensor readings
- Asynchronous data export
- Compressed historical data storage

### 7.2 Scalability Architecture

```mermaid
graph TB
    subgraph "Simulation Scaling"
        ST1[Single Node]
        ST2[Multi-Node 10]
        ST3[Multi-Node 50]
        ST4[Multi-Node 100+]
    end

    subgraph "Resource Requirements"
        MEM[Memory Usage]
        CPU[CPU Usage]
        TIME[Execution Time]
        IO[I/O Operations]
    end

    ST1 --> MEM
    ST2 --> MEM
    ST3 --> MEM
    ST4 --> MEM

    ST1 --> CPU
    ST2 --> CPU
    ST3 --> CPU
    ST4 --> CPU

    subgraph "Performance Profiles"
        P1[Baseline Performance]
        P2[Optimized Performance]
        P3[Distributed Performance]
    end

    MEM --> P1
    CPU --> P1
    TIME --> P1
    IO --> P1
```

---

## 8. Security Architecture

### 8.1 Data Protection

- **Input Validation**: All configuration parameters validated
- **Type Safety**: Strong typing for all interfaces
- **Memory Safety**: Automatic memory management in Python
- **File I/O Safety**: Secure file handling with error checking

### 8.2 Simulation Integrity

- **Deterministic Results**: Consistent output for same inputs
- **Reproducible Research**: Fixed seed values for random processes
- **Audit Trail**: Complete logging of all operations
- **Version Control**: Configuration and model versioning

---

## 9. Interface Specifications

### 9.1 Public API

```python
class SimulationAPI:
    """Public interface for simulation system"""

    def __init__(self, config_path: str = None):
        """Initialize simulation with optional configuration"""
        pass

    def create_simulation(self, duration_hours: float, **kwargs) -> str:
        """Create new simulation with specified parameters"""
        pass

    def add_node(self, node_type: str, scheduler_type: str, **params) -> None:
        """Add node to simulation"""
        pass

    def run_simulation(self) -> Dict[str, Any]:
        """Run complete simulation"""
        pass

    def get_results(self) -> Dict[str, Any]:
        """Get simulation results"""
        pass

    def export_results(self, format: str, output_path: str) -> bool:
        """Export results to specified format"""
        pass
```

### 9.2 CLI Interface

```bash
# Basic simulation
python -m iot_sleep_scheduler run --mode basic --duration 24

# Comparative analysis
python -m iot_sleep_scheduler run --mode comparative

# Custom configuration
python -m iot_sleep_scheduler run --config custom_config.json

# Scalability testing
python -m iot_sleep_scheduler run --mode scalability --nodes 50
```

---

## 10. Deployment Architecture

### 10.1 Development Environment

```mermaid
graph LR
    A[Developer Machine] --> B[Git Repository]
    B --> C[Virtual Environment]
    C --> D[Dependencies]
    D --> E[Source Code]
    E --> F[Testing Suite]
    F --> G[Documentation]
```

### 10.2 Production Deployment

```mermaid
graph TB
    subgraph "Container Environment"
        APP[Application Container]
        DATA[Data Volume]
        LOGS[Log Volume]
    end

    subgraph "External Dependencies"
        PYTHON[Python Runtime]
        LIBS[Required Libraries]
        CONFIG[Configuration Files]
    end

    APP --> DATA
    APP --> LOGS
    PYTHON --> APP
    LIBS --> APP
    CONFIG --> APP
```

---

## 11. Evolution Architecture

### 11.1 Extension Points

- **New Schedulers**: Plugin-based scheduler architecture
- **Custom Sensors**: Modular sensor framework
- **Advanced Analysis**: Extensible analysis engine
- **Visualization**: Pluggable graph generation system

### 11.2 Future Enhancements

- **Machine Learning**: Integration of ML-based optimization
- **Distributed Simulation**: Multi-node simulation across clusters
- **Real-World Integration**: Hardware-in-the-loop testing
- **Web Interface**: Browser-based configuration and visualization

---

**Document Version**: 1.0
**Last Updated**: 2025-11-25
**Next Review**: 2025-12-25