import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import examples.serving_configs  # noqa: F401
from databricks.sdk import WorkspaceClient
from serving_platform.resource_gen import write_resources

CATALOG = "workspace"  # deve bater com o default de `catalog` em databricks.yml


def _resolve_alias_version(model_name: str, config) -> int:
    full_name = f"{CATALOG}.{config.domain}_models.{model_name}"
    return WorkspaceClient().model_versions.get_by_alias(full_name, config.alias).version


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_serving.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_resources(str(output_path), resolve_alias_version=_resolve_alias_version)
    print(f"resources written to {output_path}")
