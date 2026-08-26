from pathlib import Path

import exemplo_features.features  # noqa: F401  (importa o domínio para popular o registro)
from feature_platform.resource_gen import write_job_resource

FEATURE_PLATFORM_WHEEL_URL = (
    "https://github.com/ViniciusOtoni/platform-libs/releases/download/"
    "feature-platform-v0.1.2/feature_platform-0.1.2-py3-none-any.whl"
)

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_feature_pipeline.job.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_job_resource(
        str(output_path),
        environment_dependencies=[
            "./dist/*.whl",
            FEATURE_PLATFORM_WHEEL_URL,
        ],
    )
    print(f"resource written to {output_path}")
