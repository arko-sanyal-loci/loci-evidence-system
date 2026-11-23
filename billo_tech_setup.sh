#!/usr/bin/env bash
# billo_tech_setup.sh — Mirror all arko-sanyal-loci repos into billo-tech org.
#
# Strategy: gh repo fork --org requires billo-tech to trust the source account.
# Instead we:
#   1. Create the repo in billo-tech (gh repo create --private/--public).
#   2. Clone bare from source.
#   3. Push mirror to billo-tech.
#   4. Add a 'billo' remote to any existing local clone under ~/src.
#
# To run: bash billo_tech_setup.sh
# Prerequisite: gh must be authed as a user who is an OWNER of billo-tech.
#   If you are authed as arko-sanyal-loci (the source org), log in as your
#   personal GitHub account that owns billo-tech:
#     gh auth login --hostname github.com
#   then re-run this script.
set -euo pipefail

SOURCE_ORG="arko-sanyal-loci"
TARGET_ORG="billo-tech"
LOCAL_ROOT="${HOME}/src"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

# ── 1. Verify access ──────────────────────────────────────────────────────────
echo "→ Checking billo-tech org access..."
gh api "orgs/${TARGET_ORG}" --jq '"org: \(.login)  plan: \(.plan.name // "unknown")"'

AUTHED_USER=$(gh api user --jq '.login')
echo "  Authenticated as: ${AUTHED_USER}"

# ── 2. List source repos ──────────────────────────────────────────────────────
echo "→ Listing repos under ${SOURCE_ORG}..."
REPOS=$(gh repo list "${SOURCE_ORG}" --limit 100 --json name,isPrivate,sshUrl \
    --jq '.[] | [.name, (.isPrivate | tostring), .sshUrl] | @tsv')

MIRRORED=0
SKIPPED=0

while IFS=$'\t' read -r NAME IS_PRIVATE SSH_URL; do
    TARGET_REPO="${TARGET_ORG}/${NAME}"
    VISIBILITY="--public"
    [[ "${IS_PRIVATE}" == "true" ]] && VISIBILITY="--private"

    echo ""
    echo "── ${NAME} ──"

    # Create in target org if it doesn't exist
    if gh api "repos/${TARGET_REPO}" --silent 2>/dev/null; then
        echo "  ✓ already exists in ${TARGET_ORG}"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  → Creating ${TARGET_REPO} ${VISIBILITY}..."
        gh repo create "${TARGET_REPO}" "${VISIBILITY}" \
            --description "Mirror of arko-sanyal-loci/${NAME}" 2>&1 || {
            echo "  ⚠  Create failed — may need owner role in ${TARGET_ORG}; skipping mirror."
            continue
        }

        BARE_DIR="${WORK_DIR}/${NAME}.git"
        echo "  → Cloning bare from source..."
        git clone --bare "${SSH_URL}" "${BARE_DIR}" 2>/dev/null || {
            echo "  ⚠  Bare clone failed (private repo / no SSH key?); skipping."
            continue
        }

        echo "  → Pushing mirror to billo-tech..."
        git -C "${BARE_DIR}" push --mirror \
            "git@github.com:${TARGET_REPO}.git" 2>&1 || {
            echo "  ⚠  Mirror push failed; repo created but empty."
        }

        MIRRORED=$((MIRRORED + 1))
        echo "  ✓ mirrored"
    fi

    # Add billo remote to any local clone
    LOCAL_CLONE="${LOCAL_ROOT}/${NAME}"
    if [[ -d "${LOCAL_CLONE}/.git" ]]; then
        BILLO_URL="git@github.com:${TARGET_REPO}.git"
        if git -C "${LOCAL_CLONE}" remote | grep -q "^billo$"; then
            echo "  ✓ billo remote already set in ${LOCAL_CLONE}"
        else
            git -C "${LOCAL_CLONE}" remote add billo "${BILLO_URL}"
            echo "  + added 'billo' remote to ${LOCAL_CLONE}"
        fi
    fi
done <<< "${REPOS}"

echo ""
echo "Done. Mirrored: ${MIRRORED}  Already present: ${SKIPPED}"
echo "Push updates later with:  git push billo HEAD:main"
echo ""
echo "NOTE: If you see 403 errors, ensure your gh session is logged in as"
echo "      a billo-tech org OWNER (not as arko-sanyal-loci):"
echo "        gh auth switch   # or: gh auth login"
