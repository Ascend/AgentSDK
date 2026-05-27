Port ${SFTP_PORT}
ListenAddress 0.0.0.0
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key
SyslogFacility AUTH
LogLevel INFO
PermitRootLogin yes
StrictModes yes
MaxAuthTries 3
PasswordAuthentication yes
Subsystem sftp internal-sftp
Match User node,root
    ChrootDirectory none
    ForceCommand internal-sftp
    AllowTcpForwarding no
    X11Forwarding no
    PasswordAuthentication yes
    PubkeyAuthentication no
    AuthorizedKeysFile /dev/null
    KerberosAuthentication no
    GSSAPIAuthentication no
