#!/bin/bash
lic_file="Xilinx_$( cut -d '.' -f 1 <<< `hostname` ).lic"

export LM_LICENSE_FILE=/usr/local/cs133/Xilinx/SDAccel/${lic_file}:$LM_LICENSE_FILE
source /usr/local/cs133/Xilinx/SDAccel/2015.4/settings64.sh
source /usr/local/cs133/Xilinx/SDAccel/2015.4/driver/setup.sh
export PATH=/usr/local/cs/bin:$PATH
