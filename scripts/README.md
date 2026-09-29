# Scripts

Small tools that sit outside the application. Each one explains its own usage at the top of
the file.

| Script | What it does |
| --- | --- |
| [try_endpoint.sh](try_endpoint.sh) | Sends every sample ticket in [examples/tickets.json](../examples/tickets.json) to the live endpoint and prints each decision. The quickest way for a grader to test the service |
| [replay_traffic.py](replay_traffic.py) | Replays held-out test tickets through the live endpoint and posts the true queue back as feedback, the way a support agent would. Fills the audit log so the drift report has real traffic to judge. `--drift-demo` then sends off-topic text to show the drift alarm firing |
| [publish_model.py](publish_model.py) | Publishes a model version that is already on disk to Spaces and points production at it. It is also the rollback tool, since publishing an older version moves `models/latest.json` back to it |
| [record_run.py](record_run.py) | Writes a finished training run into the `training_runs` table after the fact. Used once, for the first cloud run, which finished before the audit tables existed |
| [run_remaining.sh](run_remaining.sh), [run_longer.sh](run_longer.sh), [run_matched.sh](run_matched.sh) | Historical. The overnight queues that ran the transformer experiments one after another on the laptop GPU. Kept so those results can be reproduced, not needed to run the service |

The application's own entry points live elsewhere. The API is `src/serve/app.py`, the
training job is `src/mlops/train_pipeline.py` (started on the GPU droplet by
`deploy/train.sh`), and infrastructure helpers are in [infra/scripts](../infra/scripts).
