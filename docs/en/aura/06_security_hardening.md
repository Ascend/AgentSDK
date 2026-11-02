# Security Statement

## Public Network Address Statement

The [public network addresses](resource/AgentSDK_public_network_addresses_0000002516443057.xlsx) that exist when the Agent SDK service starts are not accessed and do not pose any risk.

## File Permission Control

When you use an API to read a file, you must ensure that you own the file and that its permissions are no more permissive than 640. This helps prevent privilege escalation and similar security issues.
Software code or programs downloaded from external sources may pose risks. You must guarantee the security of their functions.

## Communication Matrix

Currently, Agent SDK provides distributed training capabilities and supports training in single-node and multi-node scenarios, which requires network communication. PyTorch needs to use TCP for communication,
while TorchNPU uses HCCL in CANN to communicate between NPU devices. For the communication ports, see [AgentSDK 26.2.0 Communication Matrix](resource/AgentSDK_26.2.0_communication_matrix.xlsx).
You need to pay attention to and ensure the security of the communication network between nodes. You can use methods such as iptables to reduce security risks. For network security hardening, refer to [Communication Security Hardening](#communication-security-hardening).

## Communication Security Hardening

The Agent SDK distributed training service needs to communicate between devices, and the ports opened for communication listen on 0.0.0.0 by default.
To reduce security risks, you are advised to harden security for this scenario. For example, configure the firewall with iptables, restrict external access to the ports used by distributed training before distributed training starts, and clear the firewall rules after distributed training ends.

1. Reference script template for setting and removing firewall rules
    - For firewall rule setting, refer to the following script:

    ```bash
    #!/bin/bash
    set -x

    # Port number to restrict
    port={port_to_restrict}

    # Clear old rules
    iptables -D INPUT -p tcp -j {rule_name}
    iptables -F {rule_name}
    iptables -X {rule_name}

    # Create a new rule chain
    iptables -t filter -N {rule_name}

    # Set an allowlist in multi-node scenarios to allow other nodes to access the listening port of the master node
    # Add a rule in the rule chain to allow a specific IP address range
    iptables -t filter -A {rule_name} -i eth0 -p tcp --dport $port -s {allowed_external_IP} -j ACCEPT

    # Block external addresses from accessing the distributed training port
    # Add a rule in the PORT-LIMIT-RULE rule chain to reject other IP addresses
    iptables -t filter -A {rule_name} -i {NIC_to_restrict} -p tcp --dport $port -j DROP

    # Pass traffic to the rule chain
    iptables -I INPUT -p tcp -j {rule_name}
    ```

    - For firewall rule removal, refer to the following script:

    ```bash
    #!/bin/bash
    set -x
    # Clear rules
    iptables -D INPUT -p tcp -j {rule_name}
    iptables -F {rule_name}
    iptables -X {rule_name}
    ```

2. Examples of reference scripts for setting and removing firewall rules
    1. Set the firewall for a specific port. In the script, the port number is the port to restrict. For port numbers used in Agent SDK distributed training, refer to the [communication matrix information](resource/AgentSDK_26.2.0_communication_matrix.xlsx).
   The network interface card to restrict is the NIC that the server uses for distributed communication, and the IP address allowed for external access is the IP address of the distributed training server. You can view the NIC and the server IP address with ifconfig.
   In the following output, eth0 is the NIC name and 192.168.1.1 is the server IP address:

        ```bash
        # ifconfig
        eth0
            inet addr:192.168.1.1 Bcast:192.168.1.255 Mask:255.255.255.0
            inet6 addr: fe80::230:64ee:ef1a:c1a/64 Scope:Link
        ```

    2. Assume that the server master node address is 192.168.1.1, the other server that requires distributed training is 192.168.1.2, and the training port is 4002.
        - For firewall rule setting, refer to the following script:

        ```bash
        #!/bin/bash
        set -x

        # Set the listening port
        port=4002

        # Clear old rules
        iptables -D INPUT -p tcp -j PORT-LIMIT-RULE
        iptables -F PORT-LIMIT-RULE
        iptables -X PORT-LIMIT-RULE

        # Create a new PORT-LIMIT-RULE rule chain
        iptables -t filter -N PORT-LIMIT-RULE

        # Set an allowlist in multi-node scenarios to allow 192.168.1.2 to access the master node
        # Add a rule in the PORT-LIMIT-RULE rule chain to allow a specific IP address range
        iptables -t filter -A PORT-LIMIT-RULE -i eth0 -p tcp --dport $port -s 192.168.1.2 -j ACCEPT

        # Block external addresses from accessing the distributed training port
        # Add a rule in the PORT-LIMIT-RULE rule chain to reject other IP addresses
        iptables -t filter -A PORT-LIMIT-RULE -i eth0 -p tcp --dport $port -j DROP

        # Pass traffic to the PORT-LIMIT-RULE rule chain
        iptables -I INPUT -p tcp -j PORT-LIMIT-RULE
        ```

        - For firewall rule removal, refer to the following script:

        ```bash
        #!/bin/bash
        set -x
        # Clear rules
        iptables -D INPUT -p tcp -j PORT-LIMIT-RULE
        iptables -F PORT-LIMIT-RULE
        iptables -X PORT-LIMIT-RULE
        ```
