from pathlib import Path

import exemplo_monitoring.monitoring_configs  # noqa: F401  (importa o domínio para popular o registro)
from monitoring_platform.resource_gen import write_resources

MONITORING_PLATFORM_WHEEL_URL = (
    "https://github.com/ViniciusOtoni/platform-libs/releases/download/"
    "monitoring-platform-v0.1.1/monitoring_platform-0.1.1-py3-none-any.whl"
)

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_monitoring.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_resources(
        str(output_path),
        environment_dependencies=["../dist/*.whl", MONITORING_PLATFORM_WHEEL_URL],
    )
    print(f"resources written to {output_path}")
