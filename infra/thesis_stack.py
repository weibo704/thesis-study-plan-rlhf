"""AWS infrastructure for the student study.

Shape, and why this one:

    App Runner  ──reads──>  Secrets Manager  (the OpenAI key)
         │
         └──writes──>  DynamoDB  (one table: events + spend counters)

App Runner rather than Fargate because there is no VPC, no load balancer
and no subnet maths to maintain, and an ALB alone would cost more per
month than everything here put together. Lambda was the other candidate
and loses: the app imports pandas and reads parquet at start-up, so every
scale-up would hand a student a multi-second cold start.

DynamoDB on-demand rather than provisioned: a study that runs for a few
weeks at a few dozen requests a day should cost cents, and provisioned
capacity would bill for a throughput nobody uses.

Everything here is cheap to destroy and recreate except the table, which
holds the data the thesis is being written from. That one is retained on
stack deletion on purpose -- see REMOVAL_POLICY below.
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_apprunner as apprunner
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

# Caps handed to the container. The API budget is fixed and unreplaceable,
# and the study link is public, so the ceiling lives in the deployment
# rather than in a note somewhere.
CAP_PER_PARTICIPANT_PER_DAY = "12"
CAP_GLOBAL_PER_DAY = "400"


class ThesisStudyStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------- data
        # pk = P#<participant> for events, COUNTER for spend counters.
        # sk = <iso timestamp>#<event id>, so one participant's events come
        # back in order and two writes in the same second cannot collide.
        table = dynamodb.Table(
            self, "StudyEvents",
            partition_key=dynamodb.Attribute(
                name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(
                name="sk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb
            .PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True),
            # The study data IS the thesis. A stack teardown must not take
            # it with it; an orphaned table costs pennies, a lost one costs
            # the experiment.
            removal_policy=RemovalPolicy.RETAIN,
        )

        # -------------------------------------------------------- secret
        # Created empty. The key is put in afterwards with the CLI, so it
        # never passes through CDK context, the template, or git history.
        api_key = secretsmanager.Secret(
            self, "OpenAIKey",
            description="OpenAI API key for the thesis study app",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # The inspector shows the model's internals and every comparison
        # made. Generated here rather than chosen, so it is not a word
        # anyone would guess, and never written down in the repo.
        inspect_token = secretsmanager.Secret(
            self, "InspectToken",
            description="Token for /inspect on the thesis study app",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=24, exclude_punctuation=True),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # --------------------------------------------------------- image
        image = ecr_assets.DockerImageAsset(
            self, "StudyImage",
            directory="..",                       # repo root, where the Dockerfile is
            file="Dockerfile",
            platform=ecr_assets.Platform.LINUX_AMD64,   # App Runner is x86_64
            exclude=["infra/cdk.out", ".venv", ".git"],
        )

        # ----------------------------------------------------------- IAM
        # App Runner wants two roles: one to pull the image, one the running
        # container assumes. Keeping them separate is what stops the task
        # from being able to push images.
        access_role = iam.Role(
            self, "AppRunnerAccessRole",
            assumed_by=iam.ServicePrincipal("build.apprunner.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSAppRunnerServicePolicyForECRAccess")],
        )
        instance_role = iam.Role(
            self, "AppRunnerInstanceRole",
            assumed_by=iam.ServicePrincipal("tasks.apprunner.amazonaws.com"),
        )
        table.grant_read_write_data(instance_role)
        api_key.grant_read(instance_role)
        inspect_token.grant_read(instance_role)

        # ------------------------------------------------------- service
        service = apprunner.CfnService(
            self, "StudyService",
            service_name="thesis-study",
            source_configuration=apprunner.CfnService.SourceConfigurationProperty(
                authentication_configuration=apprunner.CfnService
                .AuthenticationConfigurationProperty(
                    access_role_arn=access_role.role_arn),
                auto_deployments_enabled=False,   # deploys happen via cdk deploy
                image_repository=apprunner.CfnService.ImageRepositoryProperty(
                    image_identifier=image.image_uri,
                    image_repository_type="ECR",
                    image_configuration=apprunner.CfnService
                    .ImageConfigurationProperty(
                        port="8080",
                        runtime_environment_variables=[
                            _env("THESIS_STORE", "dynamodb"),
                            _env("THESIS_TABLE", table.table_name),
                            _env("THESIS_PUBLIC", "1"),
                            _env("THESIS_CAP_PARTICIPANT", CAP_PER_PARTICIPANT_PER_DAY),
                            _env("THESIS_CAP_GLOBAL", CAP_GLOBAL_PER_DAY),
                            _env("AWS_DEFAULT_REGION", self.region),
                        ],
                        # Injected by App Runner from Secrets Manager at
                        # start-up; the value never appears in the template,
                        # the image, or the console's env var list.
                        runtime_environment_secrets=[
                            apprunner.CfnService.KeyValuePairProperty(
                                name="OPENAI_API_KEY", value=api_key.secret_arn),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="THESIS_INSPECT_TOKEN",
                                value=inspect_token.secret_arn),
                        ],
                    ),
                ),
            ),
            instance_configuration=apprunner.CfnService
            .InstanceConfigurationProperty(
                # The catalog and embeddings sit in memory in each worker.
                # 2 GB leaves room for two preloaded gunicorn workers without
                # the OOM killer getting involved mid-study.
                cpu="1 vCPU",
                memory="2 GB",
                instance_role_arn=instance_role.role_arn,
            ),
            health_check_configuration=apprunner.CfnService
            .HealthCheckConfigurationProperty(
                protocol="HTTP",
                path="/api/study/health",
                interval=10,
                timeout=5,
                healthy_threshold=1,
                unhealthy_threshold=5,
            ),
        )
        service.node.add_dependency(access_role, instance_role)

        # --------------------------------------------------------- outputs
        CfnOutput(self, "StudyUrl",
                  value=f"https://{service.attr_service_url}/study",
                  description="Hand this link to participants")
        CfnOutput(self, "HealthUrl",
                  value=f"https://{service.attr_service_url}/api/study/health?counts=1",
                  description="Liveness plus how many events have been collected")
        CfnOutput(self, "InspectCommand",
                  value=("echo \"https://$(aws apprunner describe-service --service-arn "
                         "<arn> --query Service.ServiceUrl --output text)/inspect?k=$("
                         f"aws secretsmanager get-secret-value --secret-id "
                         f"{inspect_token.secret_arn} --query SecretString --output text)\""),
                  description="Prints the private inspector URL, token included")
        CfnOutput(self, "TableName", value=table.table_name,
                  description="DynamoDB table holding the study data")
        CfnOutput(self, "SecretArn", value=api_key.secret_arn,
                  description="Put the OpenAI key here before the first deploy")
        CfnOutput(self, "SetSecretCommand",
                  value=("aws secretsmanager put-secret-value --secret-id "
                         f"{api_key.secret_arn} --secret-string 'sk-...'"),
                  description="Run this once, with your real key")


def _env(name: str, value: str) -> "apprunner.CfnService.KeyValuePairProperty":
    return apprunner.CfnService.KeyValuePairProperty(name=name, value=value)
