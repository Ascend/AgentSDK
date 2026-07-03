# FAQ

## Network Unavailable

**Symptom**

After you start Agent SDK, the following error appears.

```bash
...
socket.gaierror: [Errno -3] Temporary failure in name resolution
...
```

**Cause Analysis**

This may happen because MindSpeed-RL fails when it resolves the node IP address.

**Solution**

Check the hostname and resolve the issue by updating `/etc/hosts`.

```bash
# Check the hostname
hostname
# Update /etc/hosts
# Add the hostname of the current device after 127.0.0.1 localhost
```
