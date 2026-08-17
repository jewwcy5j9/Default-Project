"""Validate a confirmatory result file against result_schema.json.

Usage:
    python validate_result_schema.py results/p2_k3_nested_results.json
    python validate_result_schema.py --all
    python validate_result_schema.py --fixture-pass   # confirm fixture passes
    python validate_result_schema.py --fixture-fail   # confirm missing-field fixture fails
    python validate_result_schema.py --fixture-pass --fixture-fail  # both self-tests

Exit code 0 = valid; 1 = invalid; 2 = internal error. Fixture self-tests
(--fixture-pass / --fixture-fail, combinable) instead exit 0 only when every
requested fixture behaves as expected: pass fixture VALID, fail fixture
INVALID.
Registry rule: save per-fold and per-seed predictions; never only aggregates.
Plan: SOTA_FOLLOWUP_EXECUTION_PLAN.md Workstream A1.
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "result_schema.json"


def load_schema(path=None):
    path = Path(path or SCHEMA_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def validate(obj, schema):
    """Minimal draft-07 subset: type, required, properties, minItems/maxItems,
    minProperties, items, minimum, maximum, pattern, enum, minLength,
    additionalProperties, plus custom boolean keyword "simplex" (array of
    numbers must sum to 1 within 1e-6). Returns list of error strings."""
    errors = []

    def errs(node, schema_node, path):
        out = []

        # type check
        if "type" in schema_node:
            t = schema_node["type"]
            ok = False
            if t == "object" and isinstance(node, dict):
                ok = True
            elif t == "array" and isinstance(node, list):
                ok = True
            elif t == "string" and isinstance(node, str):
                ok = True
            elif t == "number" and isinstance(node, (int, float)) and not isinstance(node, bool):
                ok = True
            elif t == "integer" and isinstance(node, int) and not isinstance(node, bool):
                ok = True
            elif t == "boolean" and isinstance(node, bool):
                ok = True
            if not ok:
                out.append(f"{path}: expected {t}, got {type(node).__name__}")
                return out

        if isinstance(node, bool):
            return out  # booleans can't satisfy the remaining numeric constraints

        if isinstance(node, dict):
            if "required" in schema_node:
                for k in schema_node["required"]:
                    if k not in node:
                        out.append(f"{path}: missing required field {k!r}")
            if "minProperties" in schema_node and len(node) < schema_node["minProperties"]:
                out.append(f"{path}: fewer than minProperties={schema_node['minProperties']}")
            props = schema_node.get("properties", {})
            extra = schema_node.get("additionalProperties")
            for k, v in node.items():
                if k in props:
                    out += errs(v, props[k], f"{path}.{k}")
                elif isinstance(extra, dict):
                    out += errs(v, extra, f"{path}.{k}")
                elif extra is not True and extra is not None:
                    out.append(f"{path}: unexpected field {k!r}")

        if isinstance(node, list):
            if schema_node.get("simplex") is True:
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in node):
                    out.append(f"{path}: simplex requires numeric items")
                elif abs(sum(node) - 1.0) > 1e-6:
                    out.append(f"{path}: simplex violation, sum={sum(node)}")
            items = schema_node.get("items", {})
            for i, v in enumerate(node):
                out += errs(v, items, f"{path}[{i}]")
            if "minItems" in schema_node and len(node) < schema_node["minItems"]:
                out.append(f"{path}: fewer than minItems={schema_node['minItems']}")
            if "maxItems" in schema_node and len(node) > schema_node["maxItems"]:
                out.append(f"{path}: more than maxItems={schema_node['maxItems']}")

        if isinstance(node, (int, float)) and not isinstance(node, bool):
            if not math.isfinite(node):
                out.append(f"{path}: non-finite number {node}")
                return out
            if "minimum" in schema_node and node < schema_node["minimum"]:
                out.append(f"{path}: {node} < minimum {schema_node['minimum']}")
            if "maximum" in schema_node and node > schema_node["maximum"]:
                out.append(f"{path}: {node} > maximum {schema_node['maximum']}")

        if isinstance(node, str):
            if "pattern" in schema_node and not __import__("re").match(schema_node["pattern"], node):
                out.append(f"{path}: {node!r} does not match {schema_node['pattern']}")
            if "enum" in schema_node and node not in schema_node["enum"]:
                out.append(f"{path}: {node!r} not in {schema_node['enum']}")
            if "minLength" in schema_node and len(node) < schema_node["minLength"]:
                out.append(f"{path}: string shorter than minLength={schema_node['minLength']}")

        return out

    return errs(obj, schema, "$")


def validate_file(path, schema=None, verbose=True):
    schema = schema if schema is not None else load_schema()
    # Reject non-finite constants: json.loads would otherwise accept NaN/
    # Infinity tokens, and NaN compares False against every minimum/maximum
    # so numeric constraints would silently pass.
    def _reject(token):
        raise ValueError(f"non-finite JSON constant {token}")
    data = json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=_reject)
    # P2 stores one schema-valid confirmatory block per system under a wrapper.
    if isinstance(data, dict) and isinstance(data.get("systems"), dict):
        errors = []
        if not data["systems"]:
            errors.append("$.systems: expected at least one system block")
        for system, block in data["systems"].items():
            for error in validate(block, schema):
                if error.startswith("$"):
                    errors.append(f"$.systems.{system}{error[1:]}")
                else:
                    errors.append(f"$.systems.{system}: {error}")
    else:
        errors = validate(data, schema)
    if verbose:
        if errors:
            print(f"INVALID  {path}  ({len(errors)} error(s))")
            for e in errors[:20]:
                print(f"    {e}")
        else:
            print(f"VALID    {path}")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--all", action="store_true",
                    help="validate every results/*.json (only meaningful "
                         "when every file has the confirmatory structure; "
                         "results/ also stores manifests and exploratory "
                         "outputs that this confirmatory schema rejects)")
    ap.add_argument("--fixture-pass", action="store_true")
    ap.add_argument("--fixture-fail", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    schema = load_schema()

    if args.fixture_pass or args.fixture_fail:
        status = 0
        if args.fixture_pass:
            f = HERE / "tests" / "fixtures" / "confirmatory_fixture.json"
            if not f.exists():
                print("MISSING fixture-pass file", file=sys.stderr)
                status = 2
            elif validate_file(f, schema, verbose=not args.quiet):
                status = max(status, 1)  # expected VALID
        if args.fixture_fail:
            f = HERE / "tests" / "fixtures" / "missing_fields_fixture.json"
            if not f.exists():
                print("MISSING fixture-fail file", file=sys.stderr)
                status = 2
            elif not validate_file(f, schema, verbose=not args.quiet):
                status = max(status, 1)  # expected INVALID
        # Fixture self-tests no longer swallow positional paths / --all:
        # if none were requested, exit with the fixture status; otherwise
        # fall through and validate them too.
        if not args.paths and not args.all:
            return status

    paths = list(args.paths)
    if args.all:
        paths += sorted(str(p) for p in (HERE / "results").glob("*.json"))

    if not paths:
        if args.fixture_pass or args.fixture_fail:
            return status
        ap.print_help()
        return 2

    any_invalid = False
    for p in paths:
        try:
            errors = validate_file(p, schema, verbose=not args.quiet)
        except Exception as exc:  # noqa: BLE001 - CLI reports and continues
            print(f"ERROR    {p}: {exc}")
            any_invalid = True
            continue
        any_invalid = any_invalid or bool(errors)
    return 1 if any_invalid else 0


if __name__ == "__main__":
    sys.exit(main())
