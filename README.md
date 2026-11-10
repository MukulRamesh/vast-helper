# vast-helper

Provision GPUs on [vast.ai](https://vast.ai) from Python or the command line.

The package wraps the vast.ai REST API and the system `ssh`/`scp` binaries. It searches for an offer, creates an instance from a template, waits for the instance and SSH to come up, runs a command, pulls files back, and destroys the instance.

## Install

```sh
pip install git+https://github.com/MukulRamesh/vast-helper.git
```

Requires Python 3.10 or newer and an OpenSSH client on `PATH`.

Set the API key:

```sh
export VAST_API_KEY=vast_...
```

`VASTAI_API_KEY` also works. Create a key in the [vast.ai console](https://console.vast.ai/manage-keys/).

## Quick start

```python
from vast_helper import OfferCriteria, ProvisionConfig, VastClient, provision_instance

client = VastClient()
config = ProvisionConfig(
    template_hash="your-template-hash",
    criteria=OfferCriteria(min_vram_gb=24, max_price=0.20),
    env={"HF_TOKEN": "hf_..."},
)
info = provision_instance(client, config)

print(f"{info.gpu_name} at ${info.price_per_hour:.2f}/hr")
print(f"ssh -i ~/.ssh/id_ed25519 -p {info.port} root@{info.host}")
```

`provision_instance` waits for the instance to reach `running` and for SSH to accept a command. If either step fails, it destroys the instance unless `keep_alive=True` is set, so a failed provision does not keep billing.

### Run a command and pull results

```python
from vast_helper import managed_instance

with managed_instance(client, config) as info:
    ssh = info.ssh()
    exit_code, output = ssh.exec_streaming(
        "cd /workspace/repo && python train.py"
    )
    ssh.download_dir("/workspace/runs", "./runs")
```

The context manager destroys the instance on exit, including when the body raises.

### Attach to a running instance

```python
from vast_helper import attach_instance

info = attach_instance(client, 45863796, connect=True)
info.ssh().exec_streaming("nvidia-smi")
```

### Search without provisioning

```python
offers = client.search_offers(OfferCriteria(min_vram_gb=24, max_price=0.20))
for offer in offers:
    print(offer["id"], offer["gpu_name"], offer["dph_total"])
```

## CLI

```sh
vast-helper verify
vast-helper search --min-vram-gb 24 --max-price 0.20
vast-helper list --running
vast-helper provision --template-hash <hash> --env HF_TOKEN=hf_...
vast-helper ssh --instance-id 12345 -- nvidia-smi
vast-helper run --template-hash <hash> --download /workspace/runs -- python train.py
vast-helper destroy 12345
```

`run` provisions an instance, waits for SSH, runs the command, downloads a directory on success, and destroys the instance. Pass `--keep-alive` to leave it running. With `--instance-id`, it attaches to an existing instance and leaves that instance alone unless `--destroy` is given.

`provision` always leaves the instance running and prints the command that destroys it.

Global options go before the subcommand:

```sh
vast-helper --api-key vast_... --log-level DEBUG search
```

Put the remote command last, and use `--` if it contains flags of its own.

## API

| Name | Purpose |
|---|---|
| `VastClient` | REST calls: `check_auth`, `search_offers`, `create_instance`, `show_instance`, `list_instances`, `destroy_instance` |
| `OfferCriteria` | Search filters, including the ones applied locally |
| `provision_instance` | Search, create, wait for running, wait for SSH |
| `managed_instance` | Context manager that provisions and destroys |
| `attach_instance` | Look up an existing instance |
| `select_running_instance` | Use an ID, or the only running instance |
| `SSHSession` | `exec_streaming`, `exec_quiet`, `download_dir`, `test_connection` |
| `verify_connection` | Pre-flight auth check |
| `destroy_instance` | Destroy with retries |
| `find_running_instances` | Filter `list_instances` by status |

`ProvisionConfig` carries the template hash, environment, SSH key, timeouts, and the `wait_for_ssh` / `keep_alive` flags. `InstanceInfo` exposes `id`, `host`, `port`, `gpu_name`, `price_per_hour`, `reliability`, and `ssh()`.

## vast.ai behavior this package handles

**Port mapping.** The external SSH port is `ports["22/tcp"][0]["HostPort"]`, not the top-level `ssh_port` field. Templates that expose several ports get a contiguous block, and `ssh_port` can point at the first port in that block instead of the SSH one.

**Local filters.** The search endpoint does not reliably apply CUDA version, disk space, PCIe bandwidth, or headless filters. The client applies them after the API call.

**Direct address.** SSH uses `public_ipaddr` when present and falls back to `ssh_host`.

**Account environment variables.** Variables created in the vast.ai console are injected into every instance. Template and request variables merge on top, and neither can override an account-level variable.

**Provisioning states.** Instances move through `provisioning`, `loading`, and `running`. Port mappings can appear after the status becomes `running`, so the wait checks for both.

**Cost safety.** `provision_instance` destroys the instance on timeout, SSH failure, or interrupt unless `keep_alive=True`. `managed_instance` destroys on exit from the body.

## Template requirements

The helper assumes a template that:

- exposes SSH on port 22, mapped externally through the `ports` record
- runs an onstart script that clones the code and installs dependencies
- has your public key registered with vast.ai

The template hash is passed as `template_hash` or `--template-hash`. The helper does not create or edit templates.

## Development

```sh
git clone https://github.com/MukulRamesh/vast-helper.git
cd vast-helper
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not integration"
```

Live tests need a key:

```sh
VAST_API_KEY=vast_... pytest -m integration
```

## License

MIT
