source /usr/local/cs133/Xilinx/SDAccel/2015.4/settings64.sh

source /u/cs/class/cs133/cs133ta/setup_oclfpga.sh

source /usr/local/cs133/FCS/Merlin/settings64.sh 

lic_file="Xilinx_$( cut -d '.' -f 1 <<< `hostname` ).lic"
export LM_LICENSE_FILE=/usr/local/cs133/Xilinx/SDAccel/${lic_file}:$LM_LICENSE_FILE
