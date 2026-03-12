#!/usr/bin/env bash
# =============================================================================
# data.sh — Sample data management CLI for the inertialai-chroma example setup
#
# Commands:
#   fetch          Download ECG5000 dataset (skips if already present)
#   fetch --force  Force re-download, replacing any existing data
#   clean          Remove all downloaded data (preserves .gitkeep)
#   --help         Print this usage information
#
# Dataset: ECG5000 from the UCR Time Series Classification Archive
# Source:  https://timeseriesclassification.com/aeon-toolkit/ECG5000.zip
#
# Note: the old Downloads/ path (ECG5000.zip) returns 404. Use aeon-toolkit/.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/sample-data"
ECG_DIR="${DATA_DIR}/ecg5000"
ZIP_FILE="${DATA_DIR}/ECG5000.zip"
DOWNLOAD_URL="https://timeseriesclassification.com/aeon-toolkit/ECG5000.zip"

# Expected ARFF files after extraction
EXPECTED_FILES=("ECG5000_TRAIN.arff" "ECG5000_TEST.arff")

# ---------------------------------------------------------------------------
# Colored output helpers
# ---------------------------------------------------------------------------

_green()   { printf "\033[32m%s\033[0m\n" "$*"; }
_blue()    { printf "\033[34m%s\033[0m\n" "$*"; }
_yellow()  { printf "\033[33m%s\033[0m\n" "$*"; }
_red()     { printf "\033[31m%s\033[0m\n" "$*"; }

info()    { _blue    "[info]    $*"; }
success() { _green   "[success] $*"; }
warn()    { _yellow  "[warn]    $*"; }
error()   { _red     "[error]   $*" >&2; }

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

check_dependencies() {
    local missing=()
    for cmd in curl unzip; do
        if ! command -v "${cmd}" &>/dev/null; then
            missing+=("${cmd}")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing[*]}"
        error "Install them and retry."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Validate that a downloaded file is a zip archive (PK magic bytes)
# ---------------------------------------------------------------------------

assert_valid_zip() {
    local file="${1}"
    # ZIP files begin with the local file header signature: PK (0x50 0x4B)
    local magic
    magic="$(od -An -N2 -tx1 "${file}" | tr -d ' \n')"
    if [[ "${magic}" != "504b" ]]; then
        error "Downloaded file is not a valid zip archive (magic bytes: ${magic})."
        error "The server may have returned an error page. Check the URL:"
        error "  ${DOWNLOAD_URL}"
        rm -f "${file}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Verify extracted data integrity by checking expected files exist
# ---------------------------------------------------------------------------

verify_extraction() {
    for file in "${EXPECTED_FILES[@]}"; do
        if [[ ! -f "${ECG_DIR}/${file}" ]]; then
            error "Verification failed: expected file not found: ${ECG_DIR}/${file}"
            return 1
        fi
    done
    return 0
}

# ---------------------------------------------------------------------------
# Check if data is already present and valid
# ---------------------------------------------------------------------------

data_is_present() {
    for file in "${EXPECTED_FILES[@]}"; do
        [[ -f "${ECG_DIR}/${file}" ]] || return 1
    done
    return 0
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_fetch() {
    local force="${1:-false}"

    check_dependencies

    if data_is_present && [[ "${force}" != "true" ]]; then
        success "ECG5000 data already present in ${ECG_DIR}"
        info "Run './data.sh fetch --force' to re-download."
        exit 0
    fi

    if [[ "${force}" == "true" ]]; then
        warn "Force flag set — removing existing data before re-downloading."
        cmd_clean
    fi

    mkdir -p "${ECG_DIR}"

    info "Downloading ECG5000 dataset from ${DOWNLOAD_URL} ..."
    curl --fail --show-error --location --progress-bar \
        --output "${ZIP_FILE}" \
        "${DOWNLOAD_URL}"

    assert_valid_zip "${ZIP_FILE}"

    info "Extracting ${ZIP_FILE} ..."
    unzip -q -o "${ZIP_FILE}" -d "${ECG_DIR}"

    # The zip may extract into a nested directory; flatten if needed
    # Find ARFF files anywhere under ECG_DIR and move them up if necessary
    for file in "${EXPECTED_FILES[@]}"; do
        if [[ ! -f "${ECG_DIR}/${file}" ]]; then
            local found
            found="$(find "${ECG_DIR}" -name "${file}" -type f | head -n 1)"
            if [[ -n "${found}" ]]; then
                mv "${found}" "${ECG_DIR}/${file}"
            fi
        fi
    done

    # Remove zip after successful extraction
    rm -f "${ZIP_FILE}"

    if ! verify_extraction; then
        error "Extraction verification failed. The archive may be corrupt or the format has changed."
        exit 1
    fi

    success "ECG5000 data ready in ${ECG_DIR}"
}

cmd_clean() {
    info "Removing downloaded data from ${DATA_DIR} ..."

    # Remove zip if leftover from a failed download
    rm -f "${ZIP_FILE}"

    # Remove extracted data directory
    rm -rf "${ECG_DIR}"

    success "Sample data cleaned. Directory structure preserved."
}

cmd_help() {
    cat <<EOF

Usage: ./data.sh <command> [options]

Commands:
  fetch             Download ECG5000 dataset into sample-data/
                    Skips silently if data is already present.
  fetch --force     Delete existing data and re-download from scratch.
  clean             Remove all downloaded data files (preserves .gitkeep).
  --help            Show this help message.

Examples:
  ./data.sh fetch
  ./data.sh fetch --force
  ./data.sh clean

Dataset source:
  ${DOWNLOAD_URL}

EOF
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

main() {
    local command="${1:-}"

    case "${command}" in
        fetch)
            local force="false"
            if [[ "${2:-}" == "--force" ]]; then
                force="true"
            fi
            cmd_fetch "${force}"
            ;;
        clean)
            cmd_clean
            ;;
        --help | -h | help)
            cmd_help
            ;;
        "")
            error "No command specified."
            cmd_help
            exit 1
            ;;
        *)
            error "Unknown command: ${command}"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
