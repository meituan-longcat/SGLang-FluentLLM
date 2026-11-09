# https://raw.githubusercontent.com/ROCm/rocmProfileData/refs/heads/master/tools/rpd2tracing.py
# commit 92d13a08328625463e9ba944cece82fc5eea36e6
def rpd_to_chrome_trace(input_rpd, output_json=None, start="0%", end="100%", format="object"):
    import gzip
    import sqlite3

    if output_json is None:
        import pathlib

        output_json = pathlib.PurePath(input_rpd).with_suffix(".trace.json.gz")

    connection = sqlite3.connect(input_rpd)

    with gzip.open(output_json, "wt", encoding="utf-8") as outfile:
        _write_chrome_trace(connection, outfile, start, end, format)

    connection.close()


def _write_chrome_trace(connection, outfile, start, end, format):
    if format == "object":
        outfile.write('{"traceEvents": ')

    outfile.write("[ {}\n")

    for row in connection.execute("select distinct gpuId from rocpd_op"):
        try:
            outfile.write(',{{"name": "process_name", "ph": "M", "pid":"{}","args":{{"name":"{}"}}}}\n'.format(row[0], "GPU" + str(row[0])))
            outfile.write(f',{{"name": "process_sort_index", "ph": "M", "pid":"{row[0]}","args":{{"sort_index":"{row[0] + 1000000}"}}}}\n')
        except ValueError:
            outfile.write("")

    for row in connection.execute("select distinct pid, tid from rocpd_api"):
        try:
            outfile.write(
                ',{{"name":"thread_name","ph":"M","pid":"{}","tid":"{}","args":{{"name":"{}"}}}}\n'.format(row[0], row[1], "Hip " + str(row[1]))
            )
            outfile.write(f',{{"name":"thread_sort_index","ph":"M","pid":"{row[0]}","tid":"{row[1]}","args":{{"sort_index":"{row[1] * 2}"}}}}\n')
        except ValueError:
            outfile.write("")

    try:
        # FIXME - these aren't rendering correctly in chrome://tracing
        for row in connection.execute("select distinct pid, tid from rocpd_hsaApi"):
            try:
                outfile.write(
                    ',{{"name":"thread_name","ph":"M","pid":"{}","tid":"{}","args":{{"name":"{}"}}}}\n'.format(row[0], row[1], "HSA " + str(row[1]))
                )
                outfile.write(
                    f',{{"name":"thread_sort_index","ph":"M","pid":"{row[0]}","tid":"{row[1]}","args":{{"sort_index":"{row[1] * 2 - 1}"}}}}\n'
                )
            except ValueError:
                outfile.write("")
    except Exception as e:
        print(f"Skipping rocpd_hsaApi thread naming (table may not exist): {e}")

    rangeStringApi = ""
    rangeStringOp = ""
    rangeStringMonitor = ""
    min_time = connection.execute("select MIN(start) from rocpd_api;").fetchall()[0][0]
    max_time = connection.execute("select MAX(end) from rocpd_api;").fetchall()[0][0]
    if min_time is None:
        raise Exception("Trace file is empty.")

    print("Timestamps:")
    print(f"\t    first: \t{min_time / 1000} us")
    print(f"\t     last: \t{max_time / 1000} us")
    print(f"\t duration: \t{(max_time - min_time) / 1000000000} seconds")

    start_time = min_time / 1000
    end_time = max_time / 1000

    if start:
        start_time = ((max_time - min_time) * (int(start.replace("%", "")) / 100) + min_time) / 1000 if "%" in start else int(start)
        rangeStringApi = f"where rocpd_api.start/1000 >= {start_time}"
        rangeStringOp = f"where rocpd_op.start/1000 >= {start_time}"
        rangeStringMonitor = f"where start/1000 >= {start_time}"
    if end:
        end_time = ((max_time - min_time) * (int(end.replace("%", "")) / 100) + min_time) / 1000 if "%" in end else int(end)

        rangeStringApi = (
            rangeStringApi + f" and rocpd_api.start/1000 <= {end_time}" if start is not None else f"where rocpd_api.start/1000 <= {end_time}"
        )
        rangeStringOp = rangeStringOp + f" and rocpd_op.start/1000 <= {end_time}" if start is not None else f"where rocpd_op.start/1000 <= {end_time}"
        rangeStringMonitor = rangeStringMonitor + f" and start/1000 <= {end_time}" if start is not None else f"where start/1000 <= {end_time}"

    print(f"\nFilter: {rangeStringApi}")
    print(f"Output duration: {(end_time - start_time) / 1000000} seconds")

    # Output Ops

    op_query = (
        "select A.string as optype, B.string as description, gpuId, queueId, "
        "rocpd_op.start/1000.0, (rocpd_op.end-rocpd_op.start) / 1000.0 "
        "from rocpd_op "
        "INNER JOIN rocpd_string A on A.id = rocpd_op.opType_id "
        "INNER Join rocpd_string B on B.id = rocpd_op.description_id "
        f"{rangeStringOp}"
    )
    for row in connection.execute(op_query):
        try:
            name = row[0] if len(row[1]) == 0 else row[1]
            outfile.write(
                f',{{"pid":"{row[2]}","tid":"{row[3]}","name":"{name}","ts":"{row[4]}","dur":"{row[5]}","ph":"X","args":{{"desc":"{row[0]}"}}}}\n'
            )
        except ValueError:
            outfile.write("")

    # Output Graph executions on GPU
    try:
        graph_query = (
            "select graphExec, gpuId, queueId, min(start)/1000.0, "
            "(max(end)-min(start))/1000.0, count(*) "
            "from rocpd_graphLaunchapi A "
            "join rocpd_api_ops B on B.api_id = A.api_ptr_id "
            "join rocpd_op C on C.id = B.op_id "
            f"{rangeStringMonitor} group by api_ptr_id"
        )
        for row in connection.execute(graph_query):
            try:
                outfile.write(
                    ',{{"pid":"{}","tid":"{}","name":"{}","ts":"{}","dur":"{}","ph":"X","args":{{"kernels":"{}"}}}}\n'.format(
                        row[1], row[2], f"Graph {row[0]}", row[3], row[4], row[5]
                    )
                )
            except ValueError:
                outfile.write("")
    except Exception as e:
        print(f"Skipping rocpd_graphLaunchapi output (table may not exist): {e}")

    # Output apis
    api_query = (
        "select A.string as apiName, B.string as args, pid, tid, "
        "rocpd_api.start/1000.0, (rocpd_api.end-rocpd_api.start) / 1000.0, "
        "(rocpd_api.end != rocpd_api.start) as has_duration "
        "from rocpd_api "
        "INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id "
        "INNER Join rocpd_string B on B.id = rocpd_api.args_id "
        f"{rangeStringApi} order by rocpd_api.id"
    )
    for row in connection.execute(api_query):
        try:
            if row[0] == "UserMarker":
                if row[6] == 0:  # instantanuous "mark" messages
                    outfile.write(
                        ',{{"pid":"{}","tid":"{}","name":"{}","ts":"{}","ph":"i","s":"p","args":{{"desc":"{}"}}}}\n'.format(
                            row[2],
                            row[3],
                            row[1].replace('"', ""),
                            row[4],
                            row[1].replace('"', ""),
                        )
                    )
                else:
                    outfile.write(
                        ',{{"pid":"{}","tid":"{}","name":"{}","ts":"{}","dur":"{}","ph":"X","args":{{"desc":"{}"}}}}\n'.format(
                            row[2],
                            row[3],
                            row[1].replace('"', ""),
                            row[4],
                            row[5],
                            row[1].replace('"', ""),
                        )
                    )
            else:
                outfile.write(
                    ',{{"pid":"{}","tid":"{}","name":"{}","ts":"{}","dur":"{}","ph":"X","args":{{"desc":"{}"}}}}\n'.format(
                        row[2],
                        row[3],
                        row[0],
                        row[4],
                        row[5],
                        row[1].replace('"', "").replace("\t", ""),
                    )
                )
        except ValueError:
            outfile.write("")

    # Output api->op linkage
    api_op_query = (
        "select rocpd_api_ops.id, pid, tid, gpuId, queueId, "
        "rocpd_api.end/1000.0 - 2, rocpd_op.start/1000.0 "
        "from rocpd_api_ops "
        "INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id "
        "INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id "
        f"{rangeStringApi}"
    )
    for row in connection.execute(api_op_query):
        try:
            fromtime = row[5] if row[5] < row[6] else row[6]
            outfile.write(f',{{"pid":"{row[1]}","tid":"{row[2]}","cat":"api_op","name":"api_op","ts":"{fromtime}","id":"{row[0]}","ph":"s"}}\n')
            outfile.write(
                f',{{"pid":"{row[3]}","tid":"{row[4]}","cat":"api_op","name":"api_op","ts":"{row[6]}","id":"{row[0]}","ph":"f", "bp":"e"}}\n'
            )
        except ValueError:
            outfile.write("")

    try:
        hsa_api_query = (
            "select A.string as apiName, B.string as args, pid, tid, "
            "rocpd_hsaApi.start/1000.0, (rocpd_hsaApi.end-rocpd_hsaApi.start) / 1000.0 "
            "from rocpd_hsaApi "
            "INNER JOIN rocpd_string A on A.id = rocpd_hsaApi.apiName_id "
            "INNER Join rocpd_string B on B.id = rocpd_hsaApi.args_id "
            f"{rangeStringApi} order by rocpd_hsaApi.id"
        )
        for row in connection.execute(hsa_api_query):
            try:
                outfile.write(
                    ',{{"pid":"{}","tid":"{}","name":"{}","ts":"{}","dur":"{}","ph":"X","args":{{"desc":"{}"}}}}\n'.format(
                        row[2],
                        row[3] + 1,
                        row[0],
                        row[4],
                        row[5],
                        row[1].replace('"', ""),
                    )
                )
            except ValueError:
                outfile.write("")
    except Exception as e:
        print(f"Skipping rocpd_hsaApi output (table may not exist): {e}")

    #
    # Counters
    #

    # Counters should extend to the last event in the trace.  This means they need to have a value at Tend.
    # Figure out when that is

    T_end = 0
    for row in connection.execute("SELECT max(end)/1000 from (SELECT end from rocpd_api UNION ALL SELECT end from rocpd_op)"):
        T_end = int(row[0])
    if end:
        T_end = end_time

    # Loop over GPU for per-gpu counters
    gpuIdsPresent = []
    for row in connection.execute("SELECT DISTINCT gpuId FROM rocpd_op"):
        gpuIdsPresent.append(row[0])

    for gpuId in gpuIdsPresent:

        # Create the queue depth counter
        depth = 0
        idle = 1
        queue_depth_query = (
            "select * from ("
            'select rocpd_api.start/1000.0 as ts, "1" '
            "from rocpd_api_ops "
            "INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id "
            "INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id "
            f"AND rocpd_op.gpuId = {gpuId} {rangeStringOp} "
            'UNION ALL select rocpd_op.end/1000.0, "-1" '
            "from rocpd_api_ops "
            "INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id "
            "INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id "
            f"AND rocpd_op.gpuId = {gpuId} {rangeStringOp}"
            ") order by ts"
        )
        for row in connection.execute(queue_depth_query):
            try:
                if idle and int(row[1]) > 0:
                    idle = 0
                    outfile.write(f',{{"pid":"{gpuId}","name":"Idle","ph":"C","ts":{row[0]},"args":{{"idle":{idle}}}}}\n')
                if depth == 1 and int(row[1]) < 0:
                    idle = 1
                    outfile.write(f',{{"pid":"{gpuId}","name":"Idle","ph":"C","ts":{row[0]},"args":{{"idle":{idle}}}}}\n')
                depth = depth + int(row[1])
                outfile.write(f',{{"pid":"{gpuId}","name":"QueueDepth","ph":"C","ts":{row[0]},"args":{{"depth":{depth}}}}}\n')
            except ValueError:
                outfile.write("")
        if T_end > 0:
            outfile.write(f',{{"pid":"{gpuId}","name":"Idle","ph":"C","ts":{T_end},"args":{{"idle":{idle}}}}}\n')
            outfile.write(f',{{"pid":"{gpuId}","name":"QueueDepth","ph":"C","ts":{T_end},"args":{{"depth":{depth}}}}}\n')

    # Create SMI counters
    try:
        for row in connection.execute(f"select deviceId, monitorType, start/1000.0, value from rocpd_monitor {rangeStringMonitor}"):
            outfile.write(f',{{"pid":"{row[0]}","name":"{row[1]}","ph":"C","ts":{row[2]},"args":{{"{row[1]}":{row[3]}}}}}\n')
        # Output the endpoints of the last range
        for row in connection.execute(
            f"select distinct deviceId, monitorType, max(end)/1000.0, value from rocpd_monitor {rangeStringMonitor} group by deviceId, monitorType"
        ):
            outfile.write(f',{{"pid":"{row[0]}","name":"{row[1]}","ph":"C","ts":{row[2]},"args":{{"{row[1]}":{row[3]}}}}}\n')
    except Exception:
        print("Did not find SMI data")

    # Create the (global) memory counter
    """
    sizes = {}    # address -> size
    totalSize = 0
    exp = re.compile("^ptr\\((.*)\\)\\s+size\\((.*)\\)$")
    exp2 = re.compile("^ptr\\((.*)\\)$")
    mem_query = (
        "SELECT rocpd_api.end/1000.0 as ts, B.string, '1' "
        "FROM rocpd_api "
        "INNER JOIN rocpd_string A ON A.id=rocpd_api.apiName_id "
        "INNER JOIN rocpd_string B ON B.id=rocpd_api.args_id "
        "WHERE A.string='hipFree' "
        "UNION ALL "
        "SELECT rocpd_api.start/1000.0, B.string, '0' "
        "FROM rocpd_api "
        "INNER JOIN rocpd_string A ON A.id=rocpd_api.apiName_id "
        "INNER JOIN rocpd_string B ON B.id=rocpd_api.args_id "
        "WHERE A.string='hipMalloc' "
        "ORDER BY ts asc"
    )
    for row in connection.execute(mem_query):
        try:
            if row[2] == '0':  #malloc
                m = exp.match(row[1])
                if m:
                    size = int(m.group(2), 16)
                    totalSize = totalSize + size
                    sizes[m.group(1)] = size
                    outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(row[0],totalSize))
            else:              #free
                m = exp2.match(row[1])
                if m:
                    try:    # Sometimes free addresses are not valid or listed
                        size = sizes[m.group(1)]
                        sizes[m.group(1)] = 0
                        totalSize = totalSize - size;
                        outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(row[0],totalSize))
                    except KeyError:
                        pass
        except ValueError:
            outfile.write("")
    if T_end > 0:
        outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(T_end,totalSize))
    """

    # Create "faux calling stack frame" on gpu ops traceS
    stacks = {}  # Call stacks built from UserMarker entres.     Key is 'pid,tid'
    currentFrame = {}  # "Current GPU frame" (id, name, start, end).    Key is 'pid,tid'

    class GpuFrame:
        def __init__(self):
            self.id = 0
            self.name = ""
            self.start = 0
            self.end = 0
            self.gpus = []
            self.totalOps = 0

    # FIXME: include 'start' (in ns) so we can ORDER BY it and break ties?
    marker_frame_query = (
        "SELECT '0', start/1000.0, pid, tid, B.string as label, '','','', '' "
        "from rocpd_api "
        "INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id AND A.string = 'UserMarker' "
        "INNER JOIN rocpd_string B on B.id = rocpd_api.args_id "
        "AND rocpd_api.start/1000.0 != rocpd_api.end/1000.0 "
        f"{rangeStringApi} "
        "UNION ALL "
        "SELECT '1', end/1000.0, pid, tid, B.string as label, '','','', '' "
        "from rocpd_api "
        "INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id AND A.string = 'UserMarker' "
        "INNER JOIN rocpd_string B on B.id = rocpd_api.args_id "
        "AND rocpd_api.start/1000.0 != rocpd_api.end/1000.0 "
        f"{rangeStringApi} "
        "UNION ALL "
        "SELECT '2', rocpd_api.start/1000.0, pid, tid, '' as label, "
        "gpuId, queueId, rocpd_op.start/1000.0, rocpd_op.end/1000.0 "
        "from rocpd_api_ops "
        "INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id "
        "INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id "
        f"{rangeStringApi} "
        "ORDER BY start/1000.0 asc"
    )
    for row in connection.execute(marker_frame_query):
        try:
            key = (row[2], row[3])  # Key is 'pid,tid'
            if row[0] == "0":  # Frame start
                if key not in stacks:
                    stacks[key] = []

            elif row[0] == "1":  # Frame end
                stacks[key].pop()


            elif row[0] == "2":  # API + Op
                if key in stacks and len(stacks[key]) > 0:
                    frame = stacks[key][-1]
                    gpuFrame = None
                    if key not in currentFrame:  # First op under the current api frame
                        gpuFrame = GpuFrame()
                        gpuFrame.id = frame[0]
                        gpuFrame.name = frame[1]
                        gpuFrame.start = row[7]
                        gpuFrame.end = row[8]
                        gpuFrame.gpus.append((row[5], row[6]))
                        gpuFrame.totalOps = 1
                    else:
                        gpuFrame = currentFrame[key]
                        # Another op under the same frame -> union them (but only if they are butt together)
                        if (
                            gpuFrame.id == frame[0]
                            and gpuFrame.name == frame[1]
                            and (abs(row[7] - gpuFrame.end) < 200 or abs(gpuFrame.start - row[8]) < 200)
                        ):
                            # if gpuFrame.id == frame[0] and gpuFrame.name == frame[1]:    # Another op under the same frame -> union them
                            # if False:   # Turn off frame joining
                            if row[7] < gpuFrame.start:
                                gpuFrame.start = row[7]
                            if row[8] > gpuFrame.end:
                                gpuFrame.end = row[8]
                            if (row[5], row[6]) not in gpuFrame.gpus:
                                gpuFrame.gpus.append((row[5], row[6]))
                            gpuFrame.totalOps = gpuFrame.totalOps + 1

                        else:  # This is a new frame - dump the last and make new
                            gpuFrame = currentFrame[key]
                            for dest in gpuFrame.gpus:
                                outfile.write(
                                    ',{{"pid":"{}","tid":"{}","name":"{}","ts":"{}","dur":"{}","ph":"X","args":{{"desc":"{}"}}}}\n'.format(
                                        dest[0],
                                        dest[1],
                                        gpuFrame.name.replace('"', ""),
                                        gpuFrame.start - 1,
                                        gpuFrame.end - gpuFrame.start + 1,
                                        f"UserMarker frame: {gpuFrame.totalOps} ops",
                                    )
                                )
                            currentFrame.pop(key)

                            # make the first op under the new frame
                            gpuFrame = GpuFrame()
                            gpuFrame.id = frame[0]
                            gpuFrame.name = frame[1]
                            gpuFrame.start = row[7]
                            gpuFrame.end = row[8]
                            gpuFrame.gpus.append((row[5], row[6]))
                            gpuFrame.totalOps = 1

                    currentFrame[key] = gpuFrame

        except ValueError:
            outfile.write("")

    outfile.write("]\n")

    if format == "object":
        outfile.write("} \n")
