#!/bin/bash
sudo -Hu root yum install -y cmake && yum install -y gcc && yum install -y gcc-c++ && yum install -y libXext libSM libXrender && yum -y install mesa-libGL
