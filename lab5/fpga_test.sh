#!/bin/bash

if [ ! `command -v sdaccel` ]; then
	echo "Setup SDAccel environment"
	source /usr/local/cs133/Xilinx/SDAccel/2015.4/settings64.sh
	source /usr/local/cs133/Xilinx/SDAccel/2015.4/driver/setup.sh
	export PATH=/usr/local/cs/bin:$PATH
fi

hostName=$(hostname)
if [ "$hostName" != "lnxsrv08.seas.ucla.edu" ]; then
	echo "ERROR: You are NOT on the CS133 server and no FPGA card can be used."
	exit 1
fi

echo "Testing FPGA..."
/usr/local/cs133/alphadata_test/host.exe /usr/local/cs133/alphadata_test/mmult1.xclbin > err.log
rm sdaccel_profile_summary.csv sdaccel_profile_summary.html
res=`grep "Test passed" err.log`
if [ "$res" != "Test passed!" ]; then
	echo "ERROR: FPGA board seems not working, or someone else is using it."
	echo "Please find details in err.log."
	exit 1
fi
echo "FPGA is working fine!"
rm err.log
