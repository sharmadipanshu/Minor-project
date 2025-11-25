#!/usr/bin/env python3
"""
IoT Sleep Scheduling System
Energy-Efficient Power Optimization for IoT Sensor Nodes
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of requirements file
def read_requirements():
    """Read requirements from requirements.txt, filtering out comments and empty lines."""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name before version specifier
                    package = line.split(">=")[0].split("==")[0].split("<=")[0].strip()
                    requirements.append(line)
    return requirements

setup(
    name="iot-sleep-scheduler",
    version="1.0.0",
    author="IoT Systems Research Team",
    author_email="research@iot-systems.edu",
    description="Energy-Efficient Sleep Scheduling and Power Optimization for IoT Sensor Nodes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/iot-sleep-scheduler",
    project_urls={
        "Bug Tracker": "https://github.com/your-repo/iot-sleep-scheduler/issues",
        "Documentation": "https://github.com/your-repo/iot-sleep-scheduler/docs",
        "Source Code": "https://github.com/your-repo/iot-sleep-scheduler",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Hardware",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "reports": [
            "reportlab>=3.6.0",
            "jinja2>=3.1.0",
        ],
        "performance": [
            "memory-profiler>=0.60.0",
            "line-profiler>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iot-sleep-scheduler=src.main:main",
            "iot-simulate=src.main:simulation_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
        "docs": ["*.md", "*.png", "*.jpg", "*.svg"],
        "results": ["*.csv", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "iot",
        "sensor-networks",
        "energy-optimization",
        "sleep-scheduling",
        "wireless-sensor-networks",
        "power-management",
        "adaptive-algorithms",
        "duty-cycling",
        "simulation",
        "embedded-systems",
    ],
    license="MIT",
    platforms=["any"],
)