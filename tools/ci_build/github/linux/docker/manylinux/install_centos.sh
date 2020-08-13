#!/bin/bash
set -e

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)


if ! rpm -q --quiet epel-release ; then
  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$os_major_version.noarch.rpm
fi

echo "installing for os major version : $os_major_version"
yum install -y redhat-lsb-core expat-devel libcurl-devel tar unzip curl zlib-devel make libunwind icu aria2 rsync bzip2 git bzip2-devel

if [ "$os_major_version" == "6" ]; then
    curl -o /tmp/d.sh -sSL https://dot.net/v1/dotnet-install.sh
    chmod +x /tmp/d.sh
    /tmp/d.sh --install-dir /usr/local/dotnet -c 2.1
    rm /tmp/d.sh
fi
if [ "$os_major_version" == "7" ]; then
    # install dotnet core dependencies
    yum install -y lttng-ust openssl-libs krb5-libs libicu libuuid
    # install dotnet runtimes
    yum install -y https://packages.microsoft.com/config/centos/7/packages-microsoft-prod.rpm
    yum install -y dotnet-sdk-2.1
fi

yum install -y java-1.8.0-openjdk-devel

