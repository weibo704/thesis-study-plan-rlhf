# Screenshots

Captured from the deployed service at
`https://<id>.us-east-1.awsapprunner.com`, not from a local run, so these
show the system as participants actually met it.

| File | What it shows |
|---|---|
| `00-architecture.png` | System diagram. Editable source: `../stage1-flow.drawio` |
| `01-study-landing.jpg` | What a participant sees first, including the scope statement and the note on what is recorded |
| `02-plan-comparison.jpg` | One comparison: two generated plans with their rationales. The thin-coverage warning is visible because this query sat in the 0.49–0.57 band |
| `03-inspector-weights.jpg` | The offline prior against the weights after 52 real comparisons |
| `04-inspector-trace.jpg` | Per-comparison update trace, with the surprise value and the weights each choice moved |

The inspector screenshots are token-gated in deployment; the token is
stripped from the captured URLs.
