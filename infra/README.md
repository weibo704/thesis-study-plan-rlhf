# Deploying the student study

What this creates: an App Runner service running the study container, a
DynamoDB table for the data it collects, and a Secrets Manager entry for
the OpenAI key. No VPC, no load balancer, nothing to patch.

```
App Runner ──reads──> Secrets Manager   (OpenAI key, injected at start-up)
     │
     └──writes──────> DynamoDB          (events + spend counters)
```

## Before the first deploy

Four things, once.

**1. Docker must be running.** CDK builds the image locally and pushes it
to ECR. `docker info` should succeed — start Docker Desktop if it doesn't.

**2. AWS credentials.**

```bash
aws sts get-caller-identity
```

**3. Bootstrap the account** (once per account/region — it creates the
staging bucket CDK pushes assets to):

```bash
cd infra && npx --yes aws-cdk@2 bootstrap
```

**4. Install the CDK Python deps:**

```bash
cd infra && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Deploy

```bash
cd infra && npx --yes aws-cdk@2 deploy
```

First deploy takes roughly 10 minutes, most of it the image build and the
App Runner service coming up. It prints:

- `StudyUrl` — the link to hand to students
- `HealthUrl` — liveness, and a running count of events collected
- `SetSecretCommand` — run this next

## Put the API key in

The secret is created empty on purpose, so the key never passes through
CDK context, the CloudFormation template, or git. Take the command from
the `SetSecretCommand` output and run it with your real key:

```bash
aws secretsmanager put-secret-value \
  --secret-id <SecretArn from the outputs> \
  --secret-string 'sk-...'
```

Then restart the service so it picks the value up:

```bash
aws apprunner start-deployment --service-arn <arn from the console>
```

Until the key is set, plan generation returns a 503 and students see
"we could not build plans just now" — the app stays up, it just cannot
generate.

## Check it works

```bash
curl "$(cd infra && npx --yes aws-cdk@2 deploy --outputs-file /dev/stdout 2>/dev/null | true)"
# or simply, using the HealthUrl from the outputs:
curl 'https://<id>.awsapprunner.com/api/study/health?counts=1'
```

A healthy response looks like:

```json
{"ok": true, "store": "dynamodb", "scored": true,
 "counts": {"plans_shown": 0, "choice": 0, "reason": 0}}
```

`scored: true` means the fitted reward model was found in the image. If it
is `false`, `data/agent/reward_model_poc-v6-stratified.json` is missing —
re-run stage 17 and rebuild.

## Spend caps

The API budget is fixed and the link is public, so the ceiling is enforced
in the container rather than by watching a dashboard. Both are set in
`thesis_stack.py`:

| cap | default | what it stops |
|---|---|---|
| `THESIS_CAP_PARTICIPANT` | 12 / day | one browser running the study on a loop |
| `THESIS_CAP_GLOBAL` | 400 / day | the worst case, whatever causes it |

Only plan generation is charged — it is the one paid call a student can
trigger. Counting happens *before* the model call, because a call that is
made and then fails still costs money. On AWS the global counter is a
DynamoDB atomic increment, so two instances cannot each spend the budget.

Raise them deliberately when you know the study needs more. A spent
budget cannot be refunded; a raised cap can be lowered.

## Getting the data back out

```bash
aws dynamodb scan --table-name <TableName> --output json > study_events.json
```

Then convert to the JSONL layout the offline analysis already reads:

```bash
python -m pipeline.import_study_events study_events.json
```

## Costs

At thesis scale — a few dozen students, a few hundred generations:

| | |
|---|---|
| App Runner, 1 vCPU / 2 GB | ~$5–25/month depending on idle time |
| DynamoDB on-demand | cents |
| Secrets Manager | $0.40/month |
| ECR storage | cents |

App Runner bills provisioned memory even while idle. If the study is not
running, pause the service rather than deleting the stack:

```bash
aws apprunner pause-service --service-arn <arn>
```

## Tearing it down

```bash
cd infra && npx --yes aws-cdk@2 destroy
```

**The DynamoDB table is retained on purpose.** It holds the data the
thesis is written from, and a stack teardown must not take it with it. An
orphaned table costs pennies; a lost one costs the experiment. Delete it
by hand, once the data is exported and you are sure:

```bash
aws dynamodb delete-table --table-name <TableName>
```

## Before students touch it

Collecting feedback from students for a thesis is human-subjects research.
Northeastern requires an IRB review or an exemption determination on file
*before* data collection starts, and data gathered beforehand usually
cannot be used afterwards. The study is built to make that review easy —
no names, no emails, a browser-local pseudonymous id, and a plain
statement on the first screen of what is recorded — but the determination
still has to exist. Worth raising with your advisor this week rather than
after the first run.
