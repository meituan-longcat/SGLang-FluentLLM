#多个action关联标记，可以修改或者启动命令传入
time=pp

if [ $# -lt 1 ]; then
    echo "执行命令：bash $0 \$command"
    echo "command参数需要选择其一：start | stop | code | log | log_plog | log_result | prof | status | shell | press | result | clear | clear_log "
    echo "start: 拉起服务"
    echo "stop: kill进程"
    echo "code: 更新代码"
    echo "log: 收集全量日志和结果"
    echo "log_plog: 仅收集plog"
    echo "log_result: 仅收集结果"
    echo "prof: 仅收集prof"
    echo "status: 检查启动状态"
    echo "shell: 更新npu_test/flash26b/*.sh"
    echo "press: 上传测试脚本并拉起,如使用需提前准备测试脚本，见注释"
    echo "result: 查看测试result"
    echo "clear: 删除同一个时间戳上一次执行的残余信息,包括日志/prof目录/result结果和press精度文件"
    echo "clear_log: 删除同一个时间戳上一次执行的残余信息，仅日志"
    exit 1
fi
command=$1
timenew=$2
if [ "A$timenew" != "A" ]; then
    time=$timenew
fi
echo "task time flag: ${time}"

#按照实际修改
iplist="
10.149.225.58
10.149.227.26
10.150.25.122
10.149.232.4
10.150.19.114
10.150.34.26
10.150.36.66
10.150.2.2
10.150.7.114
10.150.8.2
10.150.5.122
10.149.249.90"

currpath=`pwd`
echo "当前目录: ${currpath}"
#如果环境要做多组ip合并，每组ip配置的第二个引号必须与最后一个ip在同一行

#按照实际修改
port=8022
#按照实际修改，密码写到文件里
script_dir=$(dirname "$0")
echo "脚本目录: $script_dir"
pwfile=$script_dir/abc.txt
#按照实际修改，p/d节点个数
pnode=1
pnode_ep_size=4 # 64卡
dnode=1
dnode_ep_size=8 # 128卡
#按照实际修改
codepath=${currpath}/..
echo "code目录: ${codepath}"
#代码路径最终应为${codepath}/fluentllm
#测试脚本路径最终应为${codepath}/client_use

#如果本脚本就到测试机器执行，则需要保证${codepath} != ${testpath}

startshell=pdmode_run.sh
#提前创建好
logdir=/workdir/npu_dev_test/chenhongluo/logs
#目标执行机器上测试代码存放路径
testpath=/workdir/npu_dev_test/chenhongluo
#目标执行机上最终会生成${testpath}/fluentllm

savepath=${logdir}/logs_${time}
mkdir -p $savepath

a=0
ips=$(echo "$iplist" | tr '\n' ',' | sed 's/^,//;s/,$//')
#多组拼接可以自己改
lastip=""

for i in $iplist
do
(
    echo ${i}
    lastip=${i}
    if [ "${command}" == "code" ]; then
        time=$(date +"%Y%m%d_%H%M%S")
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "cd /workdir;mkdir -p ${testpath};cd ${testpath};rm -rf fluentllm;"
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} ${codepath}/fluentllm root@${i}:${testpath}
    fi

    if [ "${command}" == "fakeweight" ]; then
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "cd /workdir;mkdir -p ${fa
keweightdist};"
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} ${fakeweightsource} root@${i}:${fakeweightdist}
    fi

    if [ "${command}" == "shell" ]; then
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "pip3 list | grep mooncake"
#        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} ${codepath}/fluentllm/npu_test/flash26b/*.sh root@${i}:${testpath}/fluentllm/npu_test/flash26b/
    fi

    if [ "${command}" == "start" ]; then
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "export cluster_ip_list=${ips};export Node_idx=${a};env|grep cluster_ip_list;
        export MODEL_PATH=/mnt/hdfs//zw04mlnn01/checkpoint/IA/flash_pro_sft_merge_48b_dsa_260407_it0003474_int8_nosq1;
#        wget http://ml.ftp.sankuai.com:8420/upload/public/------WebKitFormBoundary1foCGBmUzcU3dVm1/mooncake_transfer_engine-0.3.9-cp311-cp311-manylinux_2_34_x86_64.whl; python3 -m pip install mooncake_transfer_engine-0.3.9-cp311-cp311-manylinux_2_34_x86_64.whl --force-reinstall --no-deps

        # prefill pp8 tp8
        export prefill_args=\"--attn-tp-size 8 --npu-lmhead-tp-size 8 --pp-size 8 --data-parallel-size 1 --dense-tp-size 1 --expert-parallel-size 8 --chunked-prefill-size 16384 --max-prefill-tokens 16384 --mem-fraction-static 0.65 --npu-enable-oe-cpu-offload --npu-o-proj-tp-size 1 --index-k-dtype bf16 --npu-disable-kv-nz --npu-enable-sp-for-indexer --enable-mla-l1-5-cache --pp-layer-nums 4 5 5 5 5 5 5 4 --npu-disable-dsa-head-parallel \"
#        export prefill_args=\"--attn-tp-size 16 --npu-lmhead-tp-size 16 --data-parallel-size 1 --dense-tp-size 16 --expert-parallel-size 16 --chunked-prefill-size 16384  --max-prefill-tokens 16384 --mem-fraction-static 0.6 --enable-mla-l1-5-cache \"
#        export prefill_args=\"--attn-tp-size 16 --data-parallel-size 1 --dense-tp-size 16 --expert-parallel-size 16 --chunked-prefill-size 16384  --max-prefill-tokens 16384 --mem-fraction-static 0.6 --enable-mla-l1-5-cache \"

        # decode
        export decode_args=\"--attn-tp-size 8 --kvp-size 8 --npu-lmhead-tp-size 16 --data-parallel-size 16 --dense-tp-size 8 --expert-parallel-size 128 --mem-fraction-static 0.77 --npu-o-proj-tp-size 8 --index-k-dtype bf16 --npu-disable-kv-nz --enable-mla-l1-5-cache --max-running-requests 64 --torch-compile-max-bs 4 --disable-radix-cache \"
#        export decode_args=\"--attn-tp-size 8 --npu-lmhead-tp-size 16 --data-parallel-size 16 --dense-tp-size 8 --expert-parallel-size 128 --mem-fraction-static 0.77 --npu-o-proj-tp-size 8 --index-k-dtype bf16 --npu-disable-kv-nz --enable-mla-l1-5-cache --max-running-requests 64 --torch-compile-max-bs 4 \"

		    env|grep Node_idx;export ASCEND_PROCESS_LOG_PATH=${testpath}/log_${time};
		    export HCCL_DETERMINISTIC=False;
#		    export ASCEND_LAUNCH_BLOCKING=1;
		    export MAX_MODEL_LEN=1000000;

        export engine_subfix_cmd=\"--is-multi-head-eagle --draft-model-path-use-base --speculative-algorithm NEXTN --speculative-num-draft-tokens 4 --speculative-num-steps 3 --speculative-eagle-topk 1 \";
        env|grep ASCEND_PROCESS_LOG_PATH;cd ${testpath}/fluentllm;

        bash npu_test/flash26b/${startshell} ${pnode} ${pnode_ep_size} ${dnode} ${dnode_ep_size} > ../testlog_${time}.log 2>&1 &"
    fi

    if [ "${command}" == "startp" ]; then
        skip_nodes=$((dnode_ep_size * dnode))
        if [ ${a} -lt ${skip_nodes} ]; then
            echo "跳过节点 ${i} (索引 ${a})"
        else
            sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "export cluster_ip_list=${ips};export Node_idx=${a};env|grep cluster_ip_list;
            export MODEL_PATH=/mnt/hdfs//zw04mlnn01/checkpoint/IA/flash_pro_sft_merge_48b_dsa_260407_it0003474_int8_nosq1;
            # prefill pp8 tp8
            export prefill_args=\"--attn-tp-size 8 --npu-lmhead-tp-size 8 --pp-size 8 --data-parallel-size 1 --dense-tp-size 1 --expert-parallel-size 8 --chunked-prefill-size 16384 --max-prefill-tokens 16384 --mem-fraction-static 0.65 --npu-enable-oe-cpu-offload --npu-o-proj-tp-size 1 --index-k-dtype bf16 --npu-disable-kv-nz --npu-enable-sp-for-indexer --enable-mla-l1-5-cache --pp-layer-nums 4 5 5 5 5 5 5 4 --npu-disable-dsa-head-parallel \"
    #        export prefill_args=\"--attn-tp-size 16 --npu-lmhead-tp-size 16 --data-parallel-size 1 --dense-tp-size 16 --expert-parallel-size 16 --chunked-prefill-size 16384  --max-prefill-tokens 16384 --mem-fraction-static 0.6 --enable-mla-l1-5-cache \"
    #        export prefill_args=\"--attn-tp-size 16 --data-parallel-size 1 --dense-tp-size 16 --expert-parallel-size 16 --chunked-prefill-size 16384  --max-prefill-tokens 16384 --mem-fraction-static 0.6 --enable-mla-l1-5-cache \"

            # decode
            export decode_args=\"--attn-tp-size 8 --kvp-size 8 --npu-lmhead-tp-size 16 --data-parallel-size 16 --dense-tp-size 8 --expert-parallel-size 128 --mem-fraction-static 0.77 --npu-o-proj-tp-size 8 --index-k-dtype bf16 --npu-disable-kv-nz --enable-mla-l1-5-cache --max-running-requests 64 --torch-compile-max-bs 4 --disable-radix-cache \"
    #        export decode_args=\"--attn-tp-size 8 --npu-lmhead-tp-size 16 --data-parallel-size 16 --dense-tp-size 8 --expert-parallel-size 128 --mem-fraction-static 0.77 --npu-o-proj-tp-size 8 --index-k-dtype bf16 --npu-disable-kv-nz --enable-mla-l1-5-cache --max-running-requests 64 --torch-compile-max-bs 4 \"

                env|grep Node_idx;export ASCEND_PROCESS_LOG_PATH=${testpath}/log_${time};
                export HCCL_DETERMINISTIC=False;
    #		    export ASCEND_LAUNCH_BLOCKING=1;
    		    export MAX_MODEL_LEN=1000000;

            export engine_subfix_cmd=\"--is-multi-head-eagle --draft-model-path-use-base --speculative-algorithm NEXTN --speculative-num-draft-tokens 4 --speculative-num-steps 3 --speculative-eagle-topk 1 \";
            env|grep ASCEND_PROCESS_LOG_PATH;cd ${testpath}/fluentllm;
            bash npu_test/flash26b/${startshell} ${pnode} ${pnode_ep_size} ${dnode} ${dnode_ep_size} > ../testlog_${time}.log 2>&1 &"
        fi
    fi

    if [ "${command}" == "startd" ]; then
        skip_nodes=$((dnode_ep_size * dnode))
        if [ ${a} -lt ${skip_nodes} ]; then
            sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "export cluster_ip_list=${ips};export Node_idx=${a};env|grep cluster_ip_list;
            export MODEL_PATH=/mnt/hdfs//zw04mlnn01/checkpoint/IA/flash_pro_sft_merge_48b_dsa_260407_it0003474_int8_nosq1;
            # prefill pp8 tp8
            export prefill_args=\"--attn-tp-size 8 --npu-lmhead-tp-size 8 --pp-size 8 --data-parallel-size 1 --dense-tp-size 1 --expert-parallel-size 8 --chunked-prefill-size 16384 --max-prefill-tokens 16384 --mem-fraction-static 0.65 --npu-enable-oe-cpu-offload --npu-o-proj-tp-size 1 --index-k-dtype bf16 --npu-disable-kv-nz --npu-enable-sp-for-indexer --enable-mla-l1-5-cache --pp-layer-nums 4 5 5 5 5 5 5 4 --npu-disable-dsa-head-parallel \"
    #        export prefill_args=\"--attn-tp-size 16 --npu-lmhead-tp-size 16 --data-parallel-size 1 --dense-tp-size 16 --expert-parallel-size 16 --chunked-prefill-size 16384  --max-prefill-tokens 16384 --mem-fraction-static 0.6 --enable-mla-l1-5-cache \"
    #        export prefill_args=\"--attn-tp-size 16 --data-parallel-size 1 --dense-tp-size 16 --expert-parallel-size 16 --chunked-prefill-size 16384  --max-prefill-tokens 16384 --mem-fraction-static 0.6 --enable-mla-l1-5-cache \"

            # decode
            export decode_args=\"--attn-tp-size 8 --kvp-size 8 --npu-lmhead-tp-size 16 --data-parallel-size 16 --dense-tp-size 8 --expert-parallel-size 128 --mem-fraction-static 0.77 --npu-o-proj-tp-size 8 --index-k-dtype bf16 --npu-disable-kv-nz --enable-mla-l1-5-cache --max-running-requests 64 --torch-compile-max-bs 4 --disable-radix-cache \"
    #        export decode_args=\"--attn-tp-size 8 --npu-lmhead-tp-size 16 --data-parallel-size 16 --dense-tp-size 8 --expert-parallel-size 128 --mem-fraction-static 0.77 --npu-o-proj-tp-size 8 --index-k-dtype bf16 --npu-disable-kv-nz --enable-mla-l1-5-cache --max-running-requests 64 --torch-compile-max-bs 4 \"

                env|grep Node_idx;export ASCEND_PROCESS_LOG_PATH=${testpath}/log_${time};
                export HCCL_DETERMINISTIC=False;
    #		    export ASCEND_LAUNCH_BLOCKING=1;
    		    export MAX_MODEL_LEN=1000000;

            export engine_subfix_cmd=\"--is-multi-head-eagle --draft-model-path-use-base --speculative-algorithm NEXTN --speculative-num-draft-tokens 4 --speculative-num-steps 3 --speculative-eagle-topk 1 \";
            env|grep ASCEND_PROCESS_LOG_PATH;cd ${testpath}/fluentllm;
            bash npu_test/flash26b/${startshell} ${pnode} ${pnode_ep_size} ${dnode} ${dnode_ep_size} > ../testlog_${time}.log 2>&1 &"
        fi
    fi

    if [ "${command}" == "status" ]; then
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "tail ${testpath}/testlog_${time}.log;tail ${testpath}/testlog_${time}.log|grep roll|grep ready"
        sleep 0.5
    fi

    if [ "${command}" == "log" ]; then
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:${testpath}/log_${time} ${savepath}/log_${time}_${i}_${a}
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:${testpath}/testlog_${time}.log ${savepath}/testlog_${time}.log_${i}_${a}
#        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:${testpath}/fluentllm ${savepath}/fluentllm_${i}_${a}
#        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:${testpath}/result_${time}.log ${savepath}/result_${time}.log_${i}_${a}
#        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:/workdir/coredump ${savepath}/coredump_${i}_${a}
#        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 8022 root@${i} "mv /workdir/coredump/* ${testpath}"
    fi

    if [ "${command}" == "log_plog" ]; then
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:${testpath}/log_${time} ${savepath}/log_${time}_${i}_${a}
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:${testpath}/result_${time}.log ${savepath}/result_${time}.log_${i}_${a}
    fi

    if [ "${command}" == "prof" ]; then
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:${testpath}/fluentllm/prof ${savepath}/prof_${i}_${a}
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${i}:${testpath}/result_${time}.log ${savepath}/result_${time}.log_${i}_${a}
    fi

    if [ "${command}" == "stop" ]; then
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "pkill -9 python;pkill -9 sglang"
    fi

    if [ "${command}" == "clear" ]; then
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "cd ${testpath};rm -rf *_${time}*"
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "cd ${testpath}/fluentllm;rm -rf prof"
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "cd ${testpath}/fluentllm/npu_test/flash26b;rm -rf *_async_qps_*"
    fi

    if [ "${command}" == "clear_log" ]; then
        sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${i} "cd ${testpath};rm -rf *log_${time}*"
    fi

    if [ "${command}" == "other" ]; then
        sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} ${codepath}/client_use/* root@${lastip}:${testpath}
    fi
) &

a=$((a+1))
done

wait

if [ "${command}" == "press" ]; then
#提前准备测试脚本包括压测脚本和拉起的batchrun.sh，存放在${codepath}/client_use/目录下
#测试命令写到batchrun.sh中,测试参考,命令可以改，重定向方式不要变
#样例10.255.149.50
#/workdir/npu_dev_test/data/batchrun_example.sh

#执行逻辑，拷贝client_use下所有文件到最后一台机器的${testpath}下,拉起batchrun.sh
    sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} ${codepath}/client_use/* root@${lastip}:${testpath}
    sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${lastip} "cd ${testpath};nohup bash batchrun.sh ${time} >> press.log 2>&1 &"
fi

if [ "${command}" == "result" ]; then
    sshpass -f ${pwfile} ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${port} root@${lastip} "tail -n 30 ${testpath}/result_${time}.log"
fi

if [ "${command}" == "log_result" ]; then
    sshpass -f ${pwfile} scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -p -P ${port} root@${lastip}:${testpath}/result_${time}.log ${savepath}/result_${time}.log_${lastip}
fi

#export HCCL_EXEC_TIMEOUT=600;export HCCL_CONNECT_TIMEOUT=600;
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

