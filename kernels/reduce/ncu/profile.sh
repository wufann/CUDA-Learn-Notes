#ncu -o profile_report python block_all_reduce.py
#ncu --set roofline -o roofline_float4 --target-processes all python block_all_reduce.py
ncu --set roofline --section SpeedOfLight -o roofline_new --target-processes all python block_all_reduce.py
