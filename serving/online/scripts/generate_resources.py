from pathlib import Path

import exemplo_serving_online.serving_configs  # noqa: F401  (importa o domínio para popular o registro)
from databricks.sdk import WorkspaceClient
from serving_platform.resource_gen import write_resources

CATALOG = "workspace"  # deve bater com o default de `catalog` em databricks.yml
SERVING_PLATFORM_WHEEL_URL = (
    "https://github.com/ViniciusOtoni/platform-libs/releases/download/"
    "serving-platform-v0.1.1/serving_platform-0.1.1-py3-none-any.whl"
)


def _resolve_alias_version(model_name: str, config) -> int:
    full_name = f"{CATALOG}.{config.domain}_models.{model_name}"
    return WorkspaceClient().model_versions.get_by_alias(full_name, config.alias).version


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_serving.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_resources(
        str(output_path),
        resolve_alias_version=_resolve_alias_version,
        environment_dependencies=["./dist/*.whl", SERVING_PLATFORM_WHEEL_URL],
    )
    print(f"resources written to {output_path}")
