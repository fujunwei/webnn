"""Patch Bazel's auto-generated local_config_cc/BUILD to add Chromium libc++
paths to cxx_builtin_include_directories.

Bazel enforces hermetic includes: only paths declared in
cxx_builtin_include_directories may appear as absolute paths in .d files.
Since we -isystem-include Chromium's libc++ headers, we must register those
paths here, otherwise Bazel rejects every compile action.

Run once after each `bazel clean --expunge` (which regenerates local_config_cc).

Idempotent: previously-injected libc++/clang entries are stripped before re-injecting.
"""
import os
import re
import shutil
import subprocess
import sys

# ---- EDIT: paths for your machine ----
CHROMIUM_SRC = os.environ.get(
    "CHROMIUM_SRC",
    r"C:\Users\fujun\workspace\chromium\src",
)
# Where to run `bazel info output_base`. Must be a Bazel workspace.
BAZEL_WORKSPACE = os.environ.get(
    "BAZEL_WORKSPACE",
    os.path.join(CHROMIUM_SRC, r"third_party\litert\src"),
)


def detect_output_base():
    """Return Bazel's output_base for BAZEL_WORKSPACE, or None on failure.

    `bazel info output_base` prints a forward-slash path on Windows; normalize
    it to a Windows path so `os.path.join` produces a valid BUILD location.
    """
    env_override = os.environ.get("BAZEL_OUTPUT_BASE")
    if env_override:
        return env_override
    try:
        result = subprocess.run(
            ["bazel", "info", "output_base"],
            cwd=BAZEL_WORKSPACE,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(
            f"ERROR: failed to run `bazel info output_base` in {BAZEL_WORKSPACE}: {e}",
            file=sys.stderr,
        )
        return None
    path = result.stdout.strip().replace("/", "\\")
    return path


BAZEL_OUTPUT_BASE = detect_output_base()
if not BAZEL_OUTPUT_BASE:
    print(
        "Set BAZEL_OUTPUT_BASE=<path> or BAZEL_WORKSPACE=<dir> and retry.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"Bazel output_base: {BAZEL_OUTPUT_BASE}")

BUILD = os.path.join(BAZEL_OUTPUT_BASE, r"external\local_config_cc\BUILD")

LIBCXX_ENTRIES_TEMPLATE = [
    r"{cr}\third_party\libc++\src\include",
    r"{cr}\third_party\libc++abi\src\include",
    r"{cr}\buildtools\third_party\libc++",
    # Bazel misdetects the clang builtin dir (parses a hash suffix by mistake).
    # Add the real path so <stdint.h>, <vadefs.h> etc. from clang's headers pass.
    r"{cr}\third_party\llvm-build\Release+Asserts\lib\clang\24\include",
]

def escape_for_bazel(p):
    # In the BUILD file, backslashes must be doubled inside "..." strings.
    return p.replace("\\", "\\\\")

LIBCXX_ENTRIES = [
    escape_for_bazel(t.format(cr=CHROMIUM_SRC)) for t in LIBCXX_ENTRIES_TEMPLATE
]

def main():
    if not os.path.exists(BUILD):
        print(f"ERROR: {BUILD} does not exist. Run `bazel build` once first to generate it.", file=sys.stderr)
        return 1

    backup = BUILD + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(BUILD, backup)
        print(f"Backup saved: {backup}")

    with open(BUILD, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip previously-injected entries (libc++ or any clang version path).
    content = re.sub(
        r',\s*"[^"]*(?:libc\+\+|llvm-build[^"]*clang\\\\\d+)[^"]*"',
        "",
        content,
    )

    pattern = re.compile(
        r"cxx_builtin_include_directories\s*=\s*\[(?P<body>.*?)\]",
        re.DOTALL,
    )

    def repl(m):
        body = m.group("body")
        # Skip empty blocks (they belong to other unused toolchains).
        if not body.strip():
            return m.group(0)
        inject = ",\n        ".join(f'"{p}"' for p in LIBCXX_ENTRIES)
        body_stripped = body.rstrip()
        if not body_stripped.endswith(","):
            body_stripped = body_stripped + ","
        return f"cxx_builtin_include_directories = [{body_stripped}\n        {inject}]"

    new_content, n_matches = pattern.subn(repl, content)
    print(f"Rewrote {n_matches} cxx_builtin_include_directories block(s) (empty blocks skipped).")

    n_injected = new_content.count("third_party\\\\libc++\\\\src\\\\include")
    print(f"libc++ path injected in {n_injected} non-empty block(s).")

    with open(BUILD, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
