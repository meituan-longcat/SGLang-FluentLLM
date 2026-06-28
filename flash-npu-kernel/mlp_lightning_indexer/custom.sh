./build.sh
cd ./build_out
./lightning_indexer_hce_aarch64.run --quiet
export LD_LIBRARY_PATH=/home/w00806478/obp_install/0930sf/ascend-toolkit/latest/opp/vendors/lightning_indexer/op_api/lib/:${LD_LIBRARY_PATH}
cd ../test
python3 test_npu_lightning_indexer.py
