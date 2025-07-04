import io
import os
import tarfile

import boto3
import sagemaker
import sagemaker.session
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
        region,
        role=None,
        default_bucket=None,
        model_package_group_name=None,  # Choose any name
        pipeline_name=None,  # You can find your pipeline name in the Studio UI (project -> Pipelines -> name)
        base_job_prefix=None,  # Choose any name
        sagemaker_project_name=None
):
    """Gets a SageMaker ML Pipeline instance working with on CustomerChurn data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    default_bucket = "rcs-pink-paper-sagemaker"  # Bucket path
    prefix = "PPP-Forecast-MLOPs"
    source_dir = f"{prefix}-Train"
    region = 'ap-south-1'


    train_s3_uri = "s3://rcs-pink-paper-sagemaker/rossmann-sales-forecasting/train/train.csv"
    store_s3_uri = "s3://rcs-pink-paper-sagemaker/rossmann-sales-forecasting/train/store.csv"
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Defining some paramter from pipeline with some default values

    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.2xlarge")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.2xlarge")
    model_approve_status = ParameterString(name="ModelApproveStatus", default_value="PendingManualApproval")
    train_data = ParameterString(name="TrainData", default_value=train_s3_uri)
    store_data = ParameterString(name="StoreData", default_value=store_s3_uri)
    acc_threshold = ParameterFloat(name="AccThreshold", default_value=0.80)
    n_estimator = ParameterInteger(name="ModelNEstimator", default_value=12)

    ## Process step

    # Step 1: Processing Step
    sklearn_image_uri = image_uris.retrieve("sklearn", region, "1.2-1")
    script_processor = ScriptProcessor(
        image_uri=sklearn_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=base_job_prefix + "-Process",
        role=role
    )

    dest_prefix = f"s3://{default_bucket}/{prefix}/train"
    step_process = ProcessingStep(
        name=base_job_prefix + "-Process",
        processor=script_processor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=dest_prefix),
        ],
        code=os.path.join(BASE_DIR, "processing.py"),
        job_arguments=[
            "--train_s3_uri", train_s3_uri,
            "--store_s3_uri", store_s3_uri
            ]
    )

    ## Train Step

    memory_buffer = io.BytesIO()

    # Create the tar.gz file in memory
    with tarfile.open(fileobj=memory_buffer, mode="w:gz") as tar:
        # Add the train.py file (or other necessary files) to the tarball
        tar.add(os.path.join(BASE_DIR, "train.py"), arcname="train.py")

    # After creating the tarball in memory, reset the buffer's position to the beginning
    memory_buffer.seek(0)

    # Upload the tarball directly to S3
    s3_client = boto3.client("s3")
    s3_path = f"{prefix}/{source_dir}/sourcedir.tar.gz"

    # Upload to S3 from memory buffer
    s3_client.upload_fileobj(memory_buffer, default_bucket, s3_path)

    source_dir_uri = f"s3://{default_bucket}/{s3_path}"

    model_path = f"s3://{default_bucket}/{prefix}/ModelArtifacts"
    sklearn_estimator = SKLearn(
        entry_point="train.py",  # Training script
        role=role,
        source_dir=source_dir_uri,
        image_uri=sklearn_image_uri,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=model_path,
        hyperparameters={
            'n_estimator': n_estimator
        }
    )

    step_train = TrainingStep(
        name=base_job_prefix + "-Train",
        estimator=sklearn_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                s3_data_type='S3Prefix',
            ),
        },
    )

    # Eval Step

    script_eval = ScriptProcessor(
    image_uri=sklearn_image_uri,
    command=["python3"],
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name=base_job_prefix + "-Eval",
    role=role
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    step_eval = ProcessingStep(
        name=base_job_prefix + "-Eval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ],
        code=os.path.join(BASE_DIR, "evaluation.py"),
        job_arguments=["--test", train_s3_uri],
        property_files=[evaluation_report]
    )


    # Model Register step

    model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json",
        )
    )

    environment_variables = {
        "SAGEMAKER_SUBMIT_DIRECTORY": source_dir_uri,
        "SAGEMAKER_PROGRAM": "train.py",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",  # Optional: Log level
        "SAGEMAKER_REGION": region
    }

    sagemaker_model = Model(
        image_uri=sklearn_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=sagemaker_session,
        env=environment_variables
    )

    step_register = RegisterModel(
        name=f"{base_job_prefix}",
        model=sagemaker_model,  # training estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m4.2xlarge", "ml.m4.4xlarge",
                            "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
                            "ml.c4.large", "ml.c4.xlarge", "ml.c4.2xlarge", "ml.c4.4xlarge"
                            ],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approve_status,
        model_metrics=model_metrics,
    )


    ## Condition step

    # Registering the fail step
    step_fail = FailStep(
        name=base_job_prefix + "-ACC_Fail",
        error_message=Join(on=" ", values=["Execution failed due to < Accuracy", acc_threshold])
    )

    cond_lte = ConditionLessThanOrEqualTo(
        right=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            # getting evaluation data from property file, which has evaluation.json file containing metrics data, property file here used to store data in pipeline it self, otherwise we have to read evaluation json file's metrics from s3 bucket. but this one directy store that data from evalaution step to property file
            json_path="regression_metrics.r2.value"  # getting that metrics data which is store as json(dict)
        ),
        left=acc_threshold
    )

    step_cond = ConditionStep(
        name=base_job_prefix + "-ACC_Cond",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[step_fail]
    )

    pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_count,
        processing_instance_type,
        training_instance_count,
        training_instance_type,
        model_approve_status,
        train_s3_uri,
        store_s3_uri,
        acc_threshold,
        n_estimator
    ],
    steps=[step_process, step_train, step_eval, step_cond],
    )

    return pipeline
