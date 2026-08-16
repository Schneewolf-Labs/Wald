+++
title = "Deploy process"
space = "engineering"
tags = ["ops", "process"]
author = "platform-team"
+++

# Deploy process

Every service ships from `main`. There is no release branch and no manual promotion step.

## How a change reaches production

1. Open a pull request. CI runs the test suite and the linter.
2. Get one approving review from a code owner.
3. Merge to `main`. The pipeline builds an image, tags it with the commit SHA, and deploys
   to staging automatically.
4. Staging soaks for fifteen minutes while smoke tests run. If they pass, production rolls
   out one region at a time.

## Rolling back

Re-deploy the previous SHA; there is no separate rollback tool. Because every deploy is a
tagged image, the previous good version is always addressable.

Roll back first and investigate afterwards. A rollback is cheap and reversible, and an
outage spent debugging forward is not.
