#${INSTALL_DIR}/python/site-packages/bin/msopgen gen -i $HOME/sample/add_custom.json -c ai_core-<soc_version> -lan cpp -out $HOME/sample/AddCustom
export PATH=/usr/local/conda/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
export PYTHONPATH=./python:$PYTHONPATH
/usr/local/Ascend/ascend-toolkit/8.3.RC1/python/site-packages/bin/msopgen gen -i update_oe_token_table.json -c ai_core-ascend910b -lan cpp -out ./