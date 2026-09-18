# Security Statement

## Public Network Address Statement

[Public network addresses](resource/AgentSDK_public_network_addresses.xlsx) that appear during Agent SDK service startup are not accessed and do not pose security risks.

## File Permission Control

When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than `640`. This helps prevent privilege escalation and similar security issues.
Software code or programs downloaded from external sources may pose risks. Users are responsible for ensuring the security of these software components.

## Communication Matrix

Agent SDK currently provides distributed training capabilities and supports training on single-node and multi-node systems, which requires network communication. PyTorch uses TCP for communication, while TorchNPU uses HCCL in CANN for communication between NPU devices. For communication ports, see the [Agent SDK 26.1.0 Communication Matrix](resource/AgentSDK_26.1.0_Communication_Matrix.xlsx).

Users should ensure the security of the communication network between nodes. You can use methods such as `iptables` to reduce security risks. For details, see [Communication Security Hardening](#communication-security-hardening).

## Communication Security Hardening

Agent SDK distributed training services require communication between devices. By default, the communication ports listen on all interfaces. To reduce security risks, users are advised to configure firewall rules for this scenario, for example, by using `iptables`.
Before starting distributed training, restrict external access to the ports used for distributed training. After distributed training ends, remove the firewall rules.

1. Reference scripts for configuring and removing firewall rules.
    * To configure firewall rules, refer to the following script:

    ```bash
    #!/bin/bash
    set -x

    # Port to restrict
    port={port_number}

    # Remove existing rules
    iptables -D INPUT -p tcp -j {rule_name}
    iptables -F {rule_name}
    iptables -X {rule_name}

    # Create a new rule chain
    iptables -t filter -N {rule_name}

    # Configure a whitelist in a multi-node scenario to allow other nodes to access the listening port of the primary node
    # Add a rule to the {rule_name} chain to allow access from specific IP addresses
    iptables -t filter -A {rule_name} -i eth0 -p tcp --dport $port -s {allowed_external_ip} -j ACCEPT

    # Block access to the distributed training port from external addresses
    # Add a rule to the {rule_name} chain to reject access from other IP addresses
    iptables -t filter -A {rule_name} -i {restricted_nic_name} -p tcp --dport $port -j DROP

    # Direct traffic to the rule chain
    iptables -I INPUT -p tcp -j {rule_name}
    ```

    * To remove firewall rules, refer to the following script:

    ```bash
    #!/bin/bash
    set -x
    # Remove rules
    iptables -D INPUT -p tcp -j {rule_name}
    iptables -F {rule_name}
    iptables -X {rule_name}
    ```

2. Example of configuring and removing firewall rules.

    1. To configure the firewall for a specific port, set the port number in the script to the port to be restricted. For the port numbers used for distributed training by Agent SDK, see the [Communication Matrix](./resource/AgentSDK_26.1.0_Communication_Matrix.xlsx). The NIC name to be restricted is the NIC used by the server for distributed communication. The allowed external IP address is the IP address of the distributed training server. You can use `ifconfig` to view the NIC name and server IP address. In the following example, `eth0` is the NIC name and `192.168.1.1` is the server IP address:

        ```bash
        # ifconfig
        eth0
            inet addr:192.168.1.1 Bcast:192.168.1.255 Mask:255.255.255.0
            inet6 addr: fe80::230:64ee:ef1a:c1a/64 Scope:Link
        ```

    2. Assume that the primary server node has the IP address `192.168.1.1`, the other server used for distributed training has the IP address `192.168.1.2`, and the training port is `4002`.

        - To configure firewall rules, refer to the following script:

        ```bash
        #!/bin/bash
        set -x

        # Set the listening port
        port=4002

        # Remove existing rules
        iptables -D INPUT -p tcp -j PORT-LIMIT-RULE
        iptables -F PORT-LIMIT-RULE
        iptables -X PORT-LIMIT-RULE

        # Create the PORT-LIMIT-RULE chain
        iptables -t filter -N PORT-LIMIT-RULE

        # Configure a whitelist in a multi-node scenario to allow 192.168.1.2 to access the primary node
        # Add a rule to the PORT-LIMIT-RULE chain to allow access from the specified IP address
        iptables -t filter -A PORT-LIMIT-RULE -i eth0 -p tcp --dport $port -s 192.168.1.2 -j ACCEPT

        # Block access to the distributed training port from external addresses
        # Add a rule to the PORT-LIMIT-RULE chain to reject access from other IP addresses
        iptables -t filter -A PORT-LIMIT-RULE -i eth0 -p tcp --dport $port -j DROP

        # Direct traffic to the PORT-LIMIT-RULE chain
        iptables -I INPUT -p tcp -j PORT-LIMIT-RULE
        ```

        - To remove firewall rules, refer to the following script:

        ```bash
        #!/bin/bash
        set -x
        # Remove rules
        iptables -D INPUT -p tcp -j PORT-LIMIT-RULE
        iptables -F PORT-LIMIT-RULE
        iptables -X PORT-LIMIT-RULE
        ```
