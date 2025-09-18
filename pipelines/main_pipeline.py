import kfp
from kfp import dsl

# Note: In a real-world scenario, these image names would point to a container registry
# where the images built from our Dockerfiles are stored.
INGEST_IMAGE = 'ingest-component:latest'
PREPARE_IMAGE = 'prepare-component:latest'
TRAIN_IMAGE = 'train-component:latest'

@dsl.container_component
def ingest_data(
    url: str,
    raw_data: dsl.Output[dsl.Dataset],
):
    """Downloads data from a URL."""
    return dsl.ContainerSpec(
        image=INGEST_IMAGE,
        command=["python", "data_ingestion.py"],
        args=[
            "--url", url,
            "--output_path", raw_data.path,
        ]
    )

@dsl.container_component
def prepare_data(
    raw_data: dsl.Input[dsl.Dataset],
    dataset_type: str,
    processed_data: dsl.Output[dsl.Dataset],
):
    """Cleans and prepares the raw data based on its type."""
    return dsl.ContainerSpec(
        image=PREPARE_IMAGE,
        command=["python", "data_handling.py"],
        args=[
            "--input_path", raw_data.path,
            "--output_path", processed_data.path,
            "--dataset_type", dataset_type,
        ]
    )

@dsl.container_component
def train_model(
    processed_data: dsl.Input[dsl.Dataset],
    target_column: str,
    experiment_name: str,
):
    """Trains a model and logs it to MLflow, configured by parameters."""
    return dsl.ContainerSpec(
        image=TRAIN_IMAGE,
        command=["python", "train.py"],
        args=[
            "--input_path", processed_data.path,
            "--target_column", target_column,
            "--experiment_name", experiment_name,
        ]
    )

@dsl.pipeline(
    name='Generic MLOps Demo Pipeline',
    description='A generic pipeline that can be configured for different datasets.'
)
def ml_pipeline(
    data_url: str,
    dataset_type: str,
    target_column: str,
    experiment_name: str,
):
    """Defines the end-to-end ML pipeline."""

    ingest_task = ingest_data(url=data_url)

    prepare_task = prepare_data(
        raw_data=ingest_task.outputs['raw_data'],
        dataset_type=dataset_type
    )

    train_task = train_model(
        processed_data=prepare_task.outputs['processed_data'],
        target_column=target_column,
        experiment_name=experiment_name
    )

if __name__ == '__main__':
    # This compilation now creates a functional pipeline definition.
    compiler = kfp.compiler.Compiler()
    compiler.compile(
        pipeline_func=ml_pipeline,
        package_path='pipeline.yaml'
    )
    print("Functional pipeline compiled successfully to pipeline.yaml")
