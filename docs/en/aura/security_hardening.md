# Security Hardening<a name="ZH-CN_TOPIC_0000002459514612"></a>

## Security Requirements<a name="ZH-CN_TOPIC_0000002459355012"></a>

When you use the CLI to read a file, ensure that you own the file and that its permissions are no more permissive than `640`. This helps prevent privilege escalation and similar security issues.

Software code or programs downloaded from external sources may pose risks. You must guarantee the security of their functions.

## Hardening Precautions<a name="ZH-CN_TOPIC_0000002459514624"></a>

The security hardening measures listed in this document provide basic recommendations. You should re-evaluate the network security posture of the entire system based on specific service requirements. When necessary, consult industry best practices and security experts.

## OS Security Hardening<a name="ZH-CN_TOPIC_0000002492554145"></a>

### Firewall Configuration<a name="ZH-CN_TOPIC_0000002459514636"></a>

After installing the OS, if common users are configured, you can add `ALWAYS_SET_PATH yes` to the `/etc/login.defs` file to prevent unauthorized privilege escalation.

### Security Configuration for the Ray Temporary Directory<a name="ZH-CN_TOPIC_0000002507137787"></a>

To ensure security, place the Ray temporary directory in the home directory of the user who runs the application. The specific path is `~/.ray/tmp`. At the same time, prevent Ray from automatically setting the temporary directory permissions to shared.

**Procedure<a name="section86341587297"></a>**

1. Create the patch file.

    Create the `ray_noshare_patch.py` file as follows:

    ```python
    import importlib
    try:
        utils = importlib.import_module("ray._private.utils")
        def _no_share(*args, **kwargs):
            return False
        utils.try_make_directory_shared = _no_share
    except Exception:
        pass
    ```

    Create the `ray_noshare.pth` file as follows:

    ```python
    import ray_noshare_patch
    ```

2. Copy the patch files to Python `site-packages` directories, such as `/home/HwHiAiUser/.local/lib/python3.11/site-packages`.

> [!NOTICE]Note
>The current temporary directory has no automatic cleanup mechanism. If the agent runs for a long time, the temporary directory may continue to grow and eventually fill the drive. To avoid this risk, you need to clean up the `~/.ray/tmp` directory yourself.

### Setting umask<a name="ZH-CN_TOPIC_0000002492474261"></a>

Set the host umask to `0027` on the host and in containers to enhance file security.

To set umask to `0027`:

1. Log in to the server as the **root** user and edit the **/etc/profile** file.

    ```bash
    vim /etc/profile
    ```

2. Append `umask 0027` to the end of the file. Save the file and exit.
3. Make the configuration take effect.

    ```bash
    source /etc/profile
    ```

### Ownerless File Hardening<a name="ZH-CN_TOPIC_0000002459355016"></a>

Differences between Docker images and the host OS may result in a mismatch between user definitions. This can lead to the creation of ownerless files during system or container operation.

You can find ownerless files on the host or in containers by running `find / -nouser -o -nogroup`. To mitigate security risks, create corresponding users and groups based on file UIDs and GIDs, or adjust existing UIDs and GIDs to match, thereby ensuring every file has a valid owner.

### Security Hardening for the Model Saving Path<a name="ZH-CN_TOPIC_0000002459355017"></a>

During training, checkpoints are saved by default in `checkpoints/${project_name}/${experiment_name}` in the current directory. This directory contains sensitive information such as model weights and optimizer state. To ensure security, harden this path as follows:

1. Ensure that the checkpoint directory permissions are set correctly:
    - Set directory permissions to `750`.
    - Set file permissions to `640`.

2. Ensure that the directory owner is the current user to prevent access by other users.

3. Avoid symbolic links to prevent path traversal attacks.

4. Check and clean up checkpoint files that are no longer needed on a regular basis to avoid sensitive information leakage.

5. If you need to store checkpoints in another path, make sure that the target path also meets the preceding security requirements.

You can use the following commands to check and set the permissions of the checkpoint directory:

```text
# Create the checkpoint directory and set permissions
mkdir -p checkpoints/${project_name}/${experiment_name}
chmod 750 checkpoints/${project_name}/${experiment_name}
chmod 750 checkpoints/${project_name}
chmod 750 checkpoints

# Set permissions for existing checkpoint files
find checkpoints -type d -exec chmod 750 {} \;
find checkpoints -type f -exec chmod 640 {} \;

# Verify the directory owner
ls -la checkpoints/
```

## Viewing Command Operation Records<a name="ZH-CN_TOPIC_0000002479124428"></a>

Command operation logs are recorded in the system history.

**Viewing installation and uninstallation history<a name="section1220492120526"></a>**

When you log out of the system or exit a container, the system saves the command history to the `~/.bash_history` file. You can check the `.bash_history` file directly to find command records.

The command history is first cached in memory and is written to the `~/.bash_history` file only when the terminal exits normally. Run the following command to write the history records in the memory to the `.bash_history` file:

```bash
history -a
```

**Modifying the number of saved history records<a name="section56389529527"></a>**

In Linux systems, the `history` command typically saves the latest 1,000 commands by default. To modify the number of saved commands, for example, to keep only 200 commands, modify the `HISTSIZE` environment variable in the `/etc/profile` file. Use the following methods:

- Use an editor (such as Vim) to modify the file.
- Use `sed` to modify the file directly with the following command:

    `sed -i 's/^HISTSIZE=number/HISTSIZE=newNumber/' /etc/profile`, where *number* represents the original number of commands and *newNumber* represents the new number of commands. For example, to change the number of saved commands from 1,000 to 200, run the following command:

    ```bash
    sed -i 's/^HISTSIZE=1000/HISTSIZE=200/' /etc/profile
    ```

After the modification, run `source /etc/profile` to make the environment variable take effect.

**Modifying timestamps in the command history file<a name="section18178420544"></a>**

To record timestamps in the command history file, add the following configuration to `/etc/profile`:

**HISTTIMEFORMAT='%F %T '**

After adding the configuration, run `source /etc/profile` to make the environment variable take effect. After timestamps are added, the `history` command result is as follows:

```text
2025-11-08 10:47:08 agentic_rl --config-path=/home/config/agentic_parameters.yaml
2025-11-08 10:47:08 agentic_rl --config-path=/home/config/agentic_parameters.yaml
2025-11-08 14:25:58 history | grep "agentic_rl"
2025-11-08 14:26:03 history | grep "agentic_rl"
```

In addition, to record command history in a custom file, set the `HISTFILE` environment variable in `/etc/profile`, and run source /etc/profile for the change to take effect. For example:

```text
HISTDIR=~/log/AgentSDK   # Configure the file for saving command history.
HISTFILE="$HISTDIR/AgentSDK.log"
mkdir -p $HISTDIR
chmod 750 $HISTDIR
touch $HISTFILE
chmod 640 $HISTFILE
USER_IP=`who -u am i 2>/dev/null| awk '{print $NF}'|sed -e 's/[()]//g'`
if [ -z $USER_IP ]
then
  USER_IP=`hostname`
fi
export HISTTIMEFORMAT="%F %T $USER_IP:`whoami` "    # command history display format: time, IP address, username, command
PROMPT_COMMAND=' { date "+%Y-%m-%d %T - $(history 1 | { read x cmd; echo "$cmd"; })"; } >> $HISTFILE'    # Write the command history to the configured file in real time.
```

The log file path is `~/log/AgentSDK`. Ensure that the drive space is sufficient and the log file permissions are set to 640.
