# Engineering Runbook Excerpt

## Deploying a Hotfix
1. Create a branch from `main` named `hotfix/<ticket-id>`.
2. Open a PR and get one approval from the on-call engineer.
3. Merge triggers an automatic canary deploy to 5% of traffic.
4. Monitor the error-rate dashboard for 15 minutes before promoting to 100%.

## Rolling Back a Deploy
Run `deploy rollback --service <name> --to <previous-version>` from the
deploy CLI. Rollbacks complete within 2 minutes and do not require approval.

## On-Call Rotation
On-call shifts are one week long, starting Monday at 9am. The on-call
engineer is the first responder for all P1 and P2 incidents.
