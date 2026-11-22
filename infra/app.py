#!/usr/bin/env python3
"""CDK entry point. Deploy with `cdk deploy` from this directory."""
import os

import aws_cdk as cdk

from thesis_stack import ThesisStudyStack

app = cdk.App()
ThesisStudyStack(
    app, "ThesisStudyStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
    description="Student study app for the course-recommendation thesis",
)
app.synth()
