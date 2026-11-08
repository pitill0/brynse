# Brynse: independent pre-merge CI

Gitea is authoritative. GitHub provides independent CI.

## Mandatory workflow

1. Develop in an isolated Gitea branch/worktree.
2. Pass the complete local quality/security gate.
3. Review and commit the exact candidate revision.
4. Publish that SHA to the GitHub mirror branch before
   merging the Gitea pull request.
5. Verify that the mirror branch HEAD equals the Gitea
   pull request source HEAD.
6. Require successful GitHub Actions jobs for all matrix
   versions: Python 3.11, 3.12 and 3.13.
7. Confirm the successful workflow run refers to the
   approved candidate SHA, not an older branch revision.
8. Any new commit requires another independent CI run.
9. Squash merge in Gitea only after these checks.
10. Verify integrated file contents and mirror main.
11. Synchronize local main with a safe fast-forward.

The resulting squash commit normally has a different SHA.
Compare the source SHA before merging and integrated content
afterwards.

A successful workflow after merging is not a pre-merge gate.

## NO-GO

- Missing, failed or stale independent CI runs.
- Divergent Gitea and GitHub candidate revisions.
- Failed local gate, audit or packaging checks.
- Unauthorized changes in other worktrees.
- Any unresolved independent audit finding.

Do not release Brynse v0.1.2 until all findings are accepted.
