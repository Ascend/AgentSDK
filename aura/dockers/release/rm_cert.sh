#!/bin/bash

set -ex

# 577c9346a499d642763812bf14351f5e9b4b446a9543f0b1897998ef54e6cd3e
rm -f /home/miniconda3/pkgs/urllib3-2.5.0-py310hd43f75c_0/info/test/dummyserver/certs/server.crt
rm -f /home/miniconda3/pkgs/urllib3-2.5.0-py310hd43f75c_0/info/test/dummyserver/certs/server.key
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/mtls/client/client.pem
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/valid/server/server.key
rm -f /home/miniconda3/pkgs/urllib3-2.5.0-py310hd43f75c_0/info/test/test/contrib/duplicate_san.pem
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/expired/ca/ca-private.key
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/valid/ca/ca-private.key
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/mtls/client/ca/ca-private.key
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/valid/server/server.pem
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/expired/server/server.csr
rm -f /home/miniconda3/pkgs/urllib3-2.5.0-py310hd43f75c_0/info/test/dummyserver/certs/cacert.key
rm -f /home/miniconda3/pkgs/urllib3-2.5.0-py310hd43f75c_0/info/test/dummyserver/certs/cacert.pem
rm -f /home/miniconda3/lib/python3.10/ensurepip/_bundled/pip-23.0.1-py3-none-any.whl
rm -f /home/miniconda3/pkgs/python-3.10.18-hbb0f47a_0/lib/python3.10/ensurepip/_bundled/pip-23.0.1-py3-none-any.whl
rm -f /home/miniconda3/share/doc/libssh2/NEWS
rm -f /home/miniconda3/pkgs/libssh2-1.11.1-hfa2bbb0_0/share/doc/libssh2/NEWS
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/expired/ca/ca.crt
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/mtls/client/ca/ca.crt
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/valid/ca/ca.crt
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/valid/server/server.csr
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/mtls/client/client.key
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/mtls/client/client.csr
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/expired/server/server.pem
rm -f /home/miniconda3/pkgs/requests-2.32.4-py310hd43f75c_0/info/test/tests/certs/expired/server/server.key
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_x448.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_ed25519.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_x25519.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_ed448.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_pkcs7.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_serialization.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_rsa.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_ssh.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/test_ec.py
rm -f /home/miniconda3/pkgs/cryptography-45.0.3-py310h2eaaae0_0/info/test/tests/hazmat/primitives/fixtures_rsa.py
rm -f /home/miniconda3/ssl/cacert.pem
rm -f /home/miniconda3/pkgs/ca-certificates-2025.2.25-hd43f75c_0/ssl/cacert.pem
rm -f /home/miniconda3/lib/python3.10/site-packages/pip/_vendor/certifi/cacert.pem
rm -f /home/miniconda3/pkgs/pip-25.1-pyhc872135_2/site-packages/pip/_vendor/certifi/cacert.pem
rm -f /home/miniconda3/lib/python3.10/site-packages/certifi/cacert.pem
rm -f /home/miniconda3/pkgs/certifi-2025.7.14-py310hd43f75c_0/lib/python3.10/site-packages/certifi/cacert.pem

# 75c45172052daa24c561a14f3d5b8749f1432595e65fd4574fbf8bc2165c6c4f
rm -f /etc/pki/ca-trust/extracted/pem/email-ca-bundle.pem
rm -f /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
rm -f /etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt
rm -f /etc/pki/ca-trust/extracted/pem/objsign-ca-bundle.pem
rm -f /pip/_vendor/certifi/cacert.pem
rm -f /etc/pki/rpm-gpg/RPM-GPG-KEY-EulerOS
rm -f /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
rm -f /etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt
rm -f /usr/share/doc/perl-IO-Socket-SSL/example/simulate_proxy.pl

# 9d03816a8eb476cafa6426509bc752951949fe92a6ab29b5cf94919021423b22
rm -f /etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt
rm -f /etc/pki/rpm-gpg/RPM-GPG-KEY-EulerOS
rm -f /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

# a7cd9eaaee2bd083b4243a37add20fc54c41f991d76d84a0ab09d5060d8c94ee
rm -f /home/miniconda3/lib/python3.10/site-packages/grpc/_cython/_credentials/roots.pem
rm -f /home/miniconda3/lib/python3.10/site-packages/virtualenv/seed/wheels/embed/pip-25.1.1-py3-none-any.whl
rm -f /home/miniconda3/lib/python3.10/site-packages/virtualenv/seed/wheels/embed/pip-25.0.1-py3-none-any.whl

#307d4cf8989044348a0a725994b27c1ab7d51885c2ffc68f139c0c653122f993
rm -f /home/miniconda3/lib/python3.10/site-packages/cmake/data/share/cmake-4.1/Templates/Windows/Windows_TemporaryKey.pfx

rm -f /pip/_vendor/certifi/cacert.pem
