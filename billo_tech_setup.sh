#!/usr/bin/env bash
# billo_tech_setup.sh — Fork all arko-sanyal-loci repos to billo-tech org
# and add billo-tech remotes to any local clones found under ~/src
set -euo pipefail

SOURCE_ORG="arko-sanyal-loci"
TARGET_ORG="billo-tech"
LOCAL_ROOT="${HOME}/src"

# ── 1. Ensure gh is authed and billo-tech is accessible ───────────────────────
echo "→ Checking billo-tech org access..."
gh api "orgs/${TARGET_ORG}" --jq '.login' || {
    echo "ERROR: cannot reach org ${TARGET_ORG} — check 'gh auth status' and org membership."
    exit 1
}

# ── 2. List all repos in source org ───────────────────────────────────────────
echo "→ Listing repos under ${SOURCE_ORG}..."
REPOS=$(gh repo list "${SOURCE_ORG}" --limit 100 --json name,isPrivate \
    --jq '.[] | [.name, (.isPrivate | tostring)] | @tsv')

FORKED=0
SKIPPED=0

while IFS=$'\t' read -r NAME IS_PRIVATE; do
    TARGET_REPO="${TARGET_ORG}/${NAME}"

    # Check if fork already exists in target org
    if gh api "repos/${TARGET_REPO}" --silent 2>/dev/null; then
        echo "  ✓ ${TARGET_REPO} already exists — skipping fork"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  → Forking ${SOURCE_ORG}/${NAME} → ${TARGET_ORG}/${NAME}..."
        gh repo fork "${SOURCE_ORG}/${NAME}" \
            --org "${TARGET_ORG}" \
            --clone=false \
            --default-branch-only=false 2>&1 || {
            echo "  ⚠  Fork of ${NAME} failed (private repo without org access?); continuing."
        }
        FORKED=$((FORKED + 1))
    fi

    # Add billo-tech remote to any local clone
    LOCAL_CLONE="${LOCAL_ROOT}/${NAME}"
    if [[ -d "${LOCAL_CLONE}/.git" ]]; then
        BILLO_URL="git@github.com:${TARGET_ORG}/${NAME}.git"
        if git -C "${LOCAL_CLONE}" remote | grep -q "^billo$"; then
            echo "  ✓ billo remote already set in ${LOCAL_CLONE}"
        else
            git -C "${LOCAL_CLONE}" remote add billo "${BILLO_URL}"
            echo "  + added 'billo' remote to ${LOCAL_CLONE}"
        fi
    fi
done <<< "${REPOS}"

echo ""
echo "Done. Forked: ${FORKED}  Already present: ${SKIPPED}"
echo "Push to billo-tech with:  git push billo HEAD:main"
