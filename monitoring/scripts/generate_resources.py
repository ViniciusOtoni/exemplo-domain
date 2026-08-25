import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import examples.monitoring_configs  # noqa: F401
from monitoring_platform.resource_gen import write_resources

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_monitoring.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_resources(str(output_path))
    print(f"resources written to {output_path}")
