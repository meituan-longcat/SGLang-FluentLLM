1, clone fluentllm
2, cd fluentllm; rm -rf npu_test; git clone -b master ssh://git@git.sankuai.com/mptech-llmp/fluentrunner.git npu_test
3, 修改npu_test/flash26b/run_3b.sh中的机器列表 iplist
4, git pull; sh npu_test/flash26b/run_3b.sh stop; sh npu_test/flash26b/run_3b.sh code; sh npu_test/flash26b/run_3b.sh start