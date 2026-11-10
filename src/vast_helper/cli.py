"""Command-line interface for vast-helper."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from typing import Any, Sequence

from vast_helper.client import OfferCriteria, VastClient
from vast_helper.constants import (
    API_POLL_TIMEOUT_S,
    DEFAULT_MAX_PRICE,
    DEFAULT_MIN_CUDA,
    DEFAULT_MIN_DISK_GB,
    DEFAULT_MIN_PCIE_BW,
    DEFAULT_MIN_PRICE,
    DEFAULT_MIN_RELIABILITY,
    DEFAULT_MIN_VRAM_GB,
    DEFAULT_SSH_KEY,
    SSH_CONNECT_TIMEOUT_S,
)
from vast_helper.errors import VastHelperError
from vast_helper.provision import (
    InstanceInfo,
    ProvisionConfig,
    attach_instance,
    destroy_instance,
    destroy_unless_kept,
    extract_ssh_port,
    provision_instance,
    select_running_instance,
    verify_connection,
    wait_for_running,
    wait_for_ssh,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``vast-helper`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="vast-helper",
        description=(
            "Provision, connect to, and manage vast.ai GPU instances. "
            "Requires a vast.ai API key in VAST_API_KEY (or --api-key)."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="vast.ai API key (default: VAST_API_KEY env var)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )

    sub = parser.add_subparsers(dest="command_name", required=True)

    # -- verify ---------------------------------------------------------
    verify_p = sub.add_parser("verify", help="Check that the API key works")
    verify_p.set_defaults(func=_cmd_verify)

    # -- search ---------------------------------------------------------
    search_p = sub.add_parser("search", help="Search GPU offers")
    _add_criteria_args(search_p)
    search_p.add_argument("--json", action="store_true", dest="as_json")
    search_p.set_defaults(func=_cmd_search)

    # -- list -----------------------------------------------------------
    list_p = sub.add_parser("list", help="List instances")
    list_p.add_argument(
        "--running",
        action="store_true",
        help="Only show running instances",
    )
    list_p.add_argument("--json", action="store_true", dest="as_json")
    list_p.set_defaults(func=_cmd_list)

    # -- provision ------------------------------------------------------
    provision_p = sub.add_parser(
        "provision",
        help="Provision a new instance and leave it running",
    )
    provision_p.add_argument(
        "--template-hash",
        required=True,
        help="vast.ai template hash ID",
    )
    provision_p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable for the instance (repeatable)",
    )
    _add_criteria_args(provision_p)
    provision_p.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    provision_p.add_argument(
        "--no-wait-for-ssh",
        action="store_false",
        dest="wait_for_ssh",
        help="Return as soon as the instance is running",
    )
    provision_p.add_argument(
        "--keep-alive",
        action="store_true",
        help="Leave the instance running if provisioning fails",
    )
    provision_p.add_argument("--json", action="store_true", dest="as_json")
    provision_p.set_defaults(func=_cmd_provision, wait_for_ssh=True)

    # -- run ------------------------------------------------------------
    run_p = sub.add_parser(
        "run",
        help="Provision (or attach to) an instance and run a command",
    )
    run_p.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Attach to an existing instance instead of provisioning",
    )
    run_p.add_argument(
        "--template-hash",
        default=None,
        help="vast.ai template hash ID (required when provisioning)",
    )
    run_p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable for the instance (repeatable)",
    )
    _add_criteria_args(run_p)
    run_p.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    run_p.add_argument(
        "--workdir",
        default=None,
        metavar="REMOTE_DIR",
        help="Remote directory to cd into before running the command",
    )
    run_p.add_argument(
        "--download",
        default=None,
        metavar="REMOTE_DIR",
        help="Download this remote directory after a successful run",
    )
    run_p.add_argument(
        "--output",
        default="./runs",
        metavar="LOCAL_DIR",
        help="Local directory for downloaded files (default: %(default)s)",
    )
    run_p.add_argument(
        "--keep-alive",
        action="store_true",
        help="Do not destroy a provisioned instance when the run finishes",
    )
    run_p.add_argument(
        "--destroy",
        action="store_true",
        help="Destroy an attached instance (--instance-id) after the run",
    )
    run_p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run remotely; use -- before it",
    )
    run_p.set_defaults(func=_cmd_run)

    # -- ssh ------------------------------------------------------------
    ssh_p = sub.add_parser(
        "ssh",
        help="Run a command on an existing instance",
    )
    ssh_p.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Instance ID (default: auto-select the only running instance)",
    )
    ssh_p.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    ssh_p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run remotely; omit to print the ssh command",
    )
    ssh_p.set_defaults(func=_cmd_ssh)

    # -- wait -----------------------------------------------------------
    wait_p = sub.add_parser(
        "wait",
        help="Wait for an instance to be running with SSH ready",
    )
    wait_p.add_argument("instance_id", type=int)
    wait_p.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    wait_p.add_argument(
        "--timeout",
        type=float,
        default=API_POLL_TIMEOUT_S,
        help="Timeout in seconds (default: %(default)s)",
    )
    wait_p.set_defaults(func=_cmd_wait)

    # -- destroy --------------------------------------------------------
    destroy_p = sub.add_parser(
        "destroy",
        help="Destroy one or more instances",
    )
    destroy_p.add_argument("instance_ids", type=int, nargs="+")
    destroy_p.set_defaults(func=_cmd_destroy)

    return parser


def _add_criteria_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-price",
        type=float,
        default=DEFAULT_MAX_PRICE,
        help="Maximum $/hr (default: %(default)s)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=DEFAULT_MIN_PRICE,
        help="Minimum $/hr (default: %(default)s)",
    )
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=DEFAULT_MIN_VRAM_GB,
        help="Minimum GPU VRAM in GB (default: %(default)s)",
    )
    parser.add_argument(
        "--min-reliability",
        type=float,
        default=DEFAULT_MIN_RELIABILITY,
        help="Minimum host reliability 0.0-1.0 (default: %(default)s)",
    )
    parser.add_argument(
        "--min-cuda",
        type=float,
        default=DEFAULT_MIN_CUDA,
        help="Minimum host CUDA version, 0 disables (default: %(default)s)",
    )
    parser.add_argument(
        "--min-disk-gb",
        type=float,
        default=DEFAULT_MIN_DISK_GB,
        help="Minimum disk GB, 0 disables (default: %(default)s)",
    )
    parser.add_argument(
        "--min-pcie-bw",
        type=float,
        default=DEFAULT_MIN_PCIE_BW,
        help="Minimum PCIe GB/s, 0 disables (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-headed",
        action="store_true",
        help="Allow GPUs with a display attached",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of offers to return",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_verify(args: argparse.Namespace) -> int:
    check = verify_connection(api_key=args.api_key)
    if not check.api_ok:
        print("API key did not authenticate.", file=sys.stderr)
        return 1
    print("API OK:  True")
    print(f"Email:   {check.user_email}")
    print(f"Balance: ${check.balance}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    offers = _client(args).search_offers(_criteria_from_args(args))
    if args.as_json:
        print(json.dumps(offers, indent=2))
        return 0
    if not offers:
        print("No offers matched.")
        return 0

    print(
        f"{'ID':>12}  {'GPU':<24} {'VRAM':>7} {'$/hr':>7} "
        f"{'Rel':>5} {'CUDA':>5} {'Disk':>7} {'PCIe':>5}  Host"
    )
    for offer in offers:
        print(
            f"{offer.get('id', '?'):>12}  "
            f"{str(offer.get('gpu_name', '?'))[:24]:<24} "
            f"{_gb(offer.get('gpu_ram')):>7} "
            f"{_money(offer.get('dph_total')):>7} "
            f"{_rel(offer.get('reliability')):>5} "
            f"{_num(offer.get('cuda_max_good')):>5} "
            f"{_gb(offer.get('disk_space')):>7} "
            f"{_num(offer.get('pcie_bw')):>5}  "
            f"{offer.get('public_ipaddr') or offer.get('ssh_host') or '-'}"
        )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    instances = _client(args).list_instances()
    if args.running:
        instances = [
            inst for inst in instances
            if inst.get("actual_status") == "running"
        ]
    if args.as_json:
        print(json.dumps(instances, indent=2))
        return 0
    if not instances:
        print("No instances found.")
        return 0

    print(
        f"{'ID':>12}  {'STATUS':<12} {'GPU':<24} {'$/hr':>7}  "
        f"{'HOST':<20} {'PORT':>6}"
    )
    for inst in instances:
        print(
            f"{inst.get('id', '?'):>12}  "
            f"{str(inst.get('actual_status', '?'))[:12]:<12} "
            f"{str(inst.get('gpu_name', '?'))[:24]:<24} "
            f"{_money(inst.get('dph_total')):>7}  "
            f"{str(inst.get('public_ipaddr') or inst.get('ssh_host') or '-'):<20} "
            f"{extract_ssh_port(inst):>6}"
        )
    return 0


def _cmd_provision(args: argparse.Namespace) -> int:
    config = ProvisionConfig(
        template_hash=args.template_hash,
        criteria=_criteria_from_args(args),
        env=_parse_env(args.env),
        ssh_key=args.ssh_key,
        wait_for_ssh=args.wait_for_ssh,
        keep_alive=args.keep_alive,
    )
    info = provision_instance(_client(args), config)
    if args.as_json:
        print(json.dumps(_instance_payload(info), indent=2))
        return 0
    _print_instance(info)
    print(f"\nDestroy with: {info.destroy_command()}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    command = _command(args.command)
    if not command:
        print("No command given. Put it after --.", file=sys.stderr)
        return 2
    if args.instance_id is None and not args.template_hash:
        print(
            "--template-hash is required unless --instance-id is given.",
            file=sys.stderr,
        )
        return 2

    client = _client(args)
    info: InstanceInfo | None = None
    provisioned = False
    success = False

    try:
        if args.instance_id is not None:
            info = attach_instance(
                client,
                args.instance_id,
                ssh_key=args.ssh_key,
            )
        else:
            info = provision_instance(
                client,
                ProvisionConfig(
                    template_hash=args.template_hash,
                    criteria=_criteria_from_args(args),
                    env=_parse_env(args.env),
                    ssh_key=args.ssh_key,
                    keep_alive=args.keep_alive,
                ),
            )
            provisioned = True

        ssh = info.ssh()
        if not ssh.test_connection():
            wait_for_ssh(ssh)

        remote_command = command
        if args.workdir:
            remote_command = f"cd {args.workdir} && {remote_command}"

        exit_code, _ = ssh.exec_streaming(remote_command)
        success = exit_code == 0

        if success and args.download:
            ssh.download_dir(args.download, args.output)

        return exit_code
    finally:
        if info is not None:
            if provisioned:
                destroy_unless_kept(
                    client,
                    info.id,
                    args.keep_alive,
                    success=success,
                )
            elif args.destroy:
                destroy_unless_kept(
                    client,
                    info.id,
                    False,
                    success=success,
                )


def _cmd_ssh(args: argparse.Namespace) -> int:
    info = select_running_instance(
        _client(args),
        args.instance_id,
        ssh_key=args.ssh_key,
    )
    ssh = info.ssh()
    command = _command(args.command)
    if not command:
        print(ssh.ssh_command())
        return 0
    if not ssh.test_connection():
        print(
            f"SSH is not responding on {info.host}:{info.port}.",
            file=sys.stderr,
        )
        return 1
    exit_code, _ = ssh.exec_streaming(command)
    return exit_code


def _cmd_wait(args: argparse.Namespace) -> int:
    client = _client(args)
    data = wait_for_running(
        client,
        args.instance_id,
        timeout_s=args.timeout,
    )
    info = InstanceInfo.from_api(data, ssh_key=args.ssh_key)
    wait_for_ssh(
        info.ssh(),
        timeout_s=min(args.timeout, SSH_CONNECT_TIMEOUT_S),
    )
    _print_instance(info)
    return 0


def _cmd_destroy(args: argparse.Namespace) -> int:
    client = _client(args)
    ok = True
    for instance_id in args.instance_ids:
        ok = destroy_instance(client, instance_id) and ok
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(args: argparse.Namespace) -> VastClient:
    return VastClient(api_key=args.api_key)


def _criteria_from_args(args: argparse.Namespace) -> OfferCriteria:
    return OfferCriteria(
        min_vram_gb=args.min_vram_gb,
        max_price=args.max_price,
        min_price=args.min_price,
        min_reliability=args.min_reliability,
        min_cuda=args.min_cuda,
        min_disk_gb=args.min_disk_gb,
        min_pcie_bw=args.min_pcie_bw,
        require_headless=not args.allow_headed,
        limit=args.limit,
    )


def _parse_env(pairs: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise VastHelperError(
                f"Invalid --env value {pair!r}; expected KEY=VALUE."
            )
        key, value = pair.split("=", 1)
        if not key:
            raise VastHelperError(
                f"Invalid --env value {pair!r}; the key is empty."
            )
        env[key] = value
    return env


def _command(parts: list[str]) -> str:
    if parts and parts[0] == "--":
        parts = parts[1:]
    return shlex.join(parts)


def _instance_payload(info: InstanceInfo) -> dict[str, Any]:
    return {
        "id": info.id,
        "host": info.host,
        "port": info.port,
        "gpu_name": info.gpu_name,
        "price_per_hour": info.price_per_hour,
        "reliability": info.reliability,
        "ssh": info.ssh().ssh_command(),
        "destroy": info.destroy_command(),
    }


def _print_instance(info: InstanceInfo) -> None:
    print(f"Instance {info.id} ready")
    if info.gpu_name:
        print(f"  GPU:         {info.gpu_name}")
    if info.price_per_hour is not None:
        print(f"  Price:       ${info.price_per_hour:.2f}/hr")
    if info.reliability is not None:
        print(f"  Reliability: {info.reliability:.0%}")
    print(f"  SSH:         {info.ssh().ssh_command()}")


def _gb(megabytes: Any) -> str:
    try:
        return f"{float(megabytes) / 1024:.0f}G"
    except (TypeError, ValueError):
        return "?"


def _money(value: Any) -> str:
    try:
        return f"${float(value):.3f}"
    except (TypeError, ValueError):
        return "?"


def _rel(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "?"


def _num(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return "?"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``vast-helper`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(message)s",
        stream=sys.stderr,
    )

    try:
        return int(args.func(args))
    except VastHelperError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
