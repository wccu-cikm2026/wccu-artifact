#!/usr/bin/env bash
# Lightweight .env loader for WCCU shell entry points. It intentionally does
# not override variables already exported by the shell.

_wccu_find_env_file() {
  if [[ -n "${WCCU_ENV_FILE:-}" ]]; then
    printf '%s\n' "${WCCU_ENV_FILE}"
    return 0
  fi
  if [[ -n "${PCSE_ENV_FILE:-}" ]]; then
    printf '%s\n' "${PCSE_ENV_FILE}"
    return 0
  fi
  local dir
  dir="$(pwd)"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/.env" ]]; then
      printf '%s\n' "$dir/.env"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/../.." && pwd)"
  if [[ -f "${repo_root}/.env" ]]; then
    printf '%s\n' "${repo_root}/.env"
    return 0
  fi
  return 1
}

_wccu_load_env_file() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local line key value first last
    line="${raw_line#${raw_line%%[![:space:]]*}}"
    line="${line%${line##*[![:space:]]}}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key//[[:space:]]/}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    # Strip inline comments in the common unquoted case.
    first="${value:0:1}"
    if [[ "$first" != "'" && "$first" != '"' ]]; then
      value="${value%%#*}"
      value="${value%${value##*[![:space:]]}}"
      value="${value#${value%%[![:space:]]*}}"
    fi
    first="${value:0:1}"
    last="${value: -1}"
    if [[ ${#value} -ge 2 && (( "$first" == "'" && "$last" == "'" ) || ( "$first" == '"' && "$last" == '"' )) ]]; then
      value="${value:1:${#value}-2}"
    fi
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done < "$env_file"
}

if env_file="$(_wccu_find_env_file)"; then
  _wccu_load_env_file "$env_file"
fi
