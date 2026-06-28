/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file op_tiling.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_TILING_H_
#define CANN_OPS_BUILT_IN_OP_TILING_H_

#include <map>
#include <chrono>
#include <memory>
#include <string>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "register/op_tiling_registry.h"
#include "platform/platform_infos_def.h"
#include "op_log.h"
#include "graph/debug/ge_log.h"
#include "lock.h"

namespace optiling {
const static bool prof_switch = std::getenv("OPTILING_PROF") != nullptr;

struct ParsedOpCompileInfo {
    std::string value;
    std::shared_ptr<void> parsed_object;
};

struct ParsedOpCompileInfoV2 {
    ge::AscendString value;
    std::shared_ptr<void> parsed_object;
};

#define REGISTER_OP_TILING_FUNC_BUFFERED(optype, opfunc)                                                              \
bool g_##optype##_TilingEntry(const TeOpParas& para, const OpCompileInfo& cinfo, OpRunInfo& rinfo) {                  \
    std::chrono::time_point<std::chrono::steady_clock> start_tiling, before_tiling, after_tiling;                     \
    if (prof_switch) {                                                                                                \
        start_tiling = std::chrono::steady_clock::now();                                                              \
    }                                                                                                                 \
    static std::map<std::string, std::shared_ptr<ParsedOpCompileInfo>> parsed_compile_info_storage;                   \
    static RWLock rwlock;                                                                                             \
    const std::string& hash_key = cinfo.key;                                                                          \
    rwlock.rdlock();                                                                                                  \
    auto found_iterator = parsed_compile_info_storage.find(hash_key);                                                 \
    std::shared_ptr<ParsedOpCompileInfo> parsed_compile_info = found_iterator != parsed_compile_info_storage.end() ?  \
    found_iterator->second : std::shared_ptr<ParsedOpCompileInfo>(nullptr);                                           \
    rwlock.unlock();                                                                                                  \
    if (!hash_key.empty() && parsed_compile_info != nullptr) {                                                        \
        std::shared_ptr<void> parsed_object_ptr = parsed_compile_info->parsed_object;                                 \
        nlohmann::json* parsed_object_of_ptr = static_cast<nlohmann::json*>(parsed_object_ptr.get());                 \
        if (prof_switch) {                                                                                            \
            before_tiling = std::chrono::steady_clock::now();                                                         \
        }                                                                                                             \
        bool res = opfunc(para.op_type, para, *parsed_object_of_ptr, rinfo);                                          \
        if (prof_switch) {                                                                                            \
            after_tiling = std::chrono::steady_clock::now();                                                          \
            uint64_t t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - start_tiling).count(); \
            uint64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>(                                      \
                after_tiling - before_tiling).count();                                                                \
            GEEVENT("[OPTILING_PROF] Found! op_name: %s, total_us: %d, tiling_us: %d", para.op_type.c_str(), t0, t1); \
        }                                                                                                             \
        return res;                                                                                                   \
    }                                                                                                                 \
    const std::string& cinfo_str = cinfo.str;                                                                         \
    {                                                                                                                 \
        std::shared_ptr<nlohmann::json> parsed_object_cinfo = std::make_shared<nlohmann::json>(                       \
            nlohmann::json::parse(cinfo_str));                                                                        \
        if (!hash_key.empty()) {                                                                                      \
            std::shared_ptr<ParsedOpCompileInfo> parsed_op_compile_info = std::make_shared<ParsedOpCompileInfo>();    \
            parsed_op_compile_info->value = cinfo_str;                                                                \
            parsed_op_compile_info->parsed_object = std::static_pointer_cast<void>(parsed_object_cinfo);              \
            rwlock.wrlock();                                                                                          \
            parsed_compile_info_storage.emplace(hash_key, parsed_op_compile_info);                                    \
            rwlock.unlock();                                                                                          \
        }                                                                                                             \
        if (prof_switch) {                                                                                            \
            before_tiling = std::chrono::steady_clock::now();                                                         \
        }                                                                                                             \
        bool ret = opfunc(para.op_type, para, *parsed_object_cinfo, rinfo);                                           \
        if (prof_switch) {                                                                                            \
            after_tiling = std::chrono::steady_clock::now();                                                          \
            uint64_t t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - start_tiling).count(); \
            uint64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count();\
            GEEVENT("[OPTILING_PROF] op_name: %s, total_us: %d, tiling_us: %d", para.op_type.c_str(), t0, t1);        \
        }                                                                                                             \
        return ret;                                                                                                   \
    }                                                                                                                 \
}                                                                                                                     \
REGISTER_OP_TILING(optype, g_##optype##_TilingEntry)


// New optiling interfaces
namespace utils {
#define REGISTER_OP_TILING_FUNC_BUFFERED_V2(optype, opfunc)                                                           \
bool g_##optype##_TilingEntry_V2(const ge::Operator& para, const optiling::utils::OpCompileInfo& cinfo,               \
                                 optiling::utils::OpRunInfo& rinfo) {                                                 \
    std::chrono::time_point<std::chrono::steady_clock> start_tiling, before_tiling, after_tiling;                     \
    if (optiling::prof_switch) {                                                                                      \
        start_tiling = std::chrono::steady_clock::now();                                                              \
    }                                                                                                                 \
    static std::map<std::string, std::shared_ptr<optiling::ParsedOpCompileInfoV2>> parsed_compile_info_storage;       \
    static RWLock rwlock;                                                                                             \
    const auto& hash_key = cinfo.GetKey();                                                                            \
    rwlock.rdlock();                                                                                                  \
    auto found_iterator = parsed_compile_info_storage.find(hash_key.GetString());                                     \
    std::shared_ptr<optiling::ParsedOpCompileInfoV2> parsed_compile_info_v2 =                                         \
        found_iterator != parsed_compile_info_storage.end() ?                                                         \
            found_iterator->second : std::shared_ptr<optiling::ParsedOpCompileInfoV2>(nullptr);                       \
    rwlock.unlock();                                                                                                  \
    ge::AscendString op_type;                                                                                         \
    ge::graphStatus ret = para.GetOpType(op_type);                                                                    \
    if (ret != ge::GRAPH_SUCCESS || op_type.GetString() == nullptr) {                                                 \
        return ge::GRAPH_FAILED;                                                                                      \
    }                                                                                                                 \
    if (hash_key.GetString() != nullptr && parsed_compile_info_v2 != nullptr) {                                       \
        std::shared_ptr<void> parsed_object_ptr = parsed_compile_info_v2->parsed_object;                              \
        nlohmann::json* parsed_object_of_ptr_v2 = static_cast<nlohmann::json*>(parsed_object_ptr.get());              \
        if (optiling::prof_switch) {                                                                                  \
            before_tiling = std::chrono::steady_clock::now();                                                         \
        }                                                                                                             \
        bool res_v2 = opfunc(op_type.GetString(), para, *parsed_object_of_ptr_v2, rinfo);                             \
        if (optiling::prof_switch) {                                                                                  \
            after_tiling = std::chrono::steady_clock::now();                                                          \
            uint64_t t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - start_tiling).count(); \
            uint64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>(                                      \
                after_tiling - before_tiling).count();                                                                \
            GEEVENT("[OPTILING_PROF] Found! op_type: %s, total_us: %d, tiling_us: %d", op_type.GetString(), t0, t1);  \
        }                                                                                                             \
        return res_v2;                                                                                                \
    }                                                                                                                 \
    const auto& cinfo_str = cinfo.GetValue();                                                                         \
    std::shared_ptr<nlohmann::json> parsed_object_cinfo_v2 = std::make_shared<nlohmann::json>(                        \
        nlohmann::json::parse(cinfo_str.GetString()));                                                                \
    if (hash_key.GetString() != nullptr) {                                                                            \
        std::shared_ptr<optiling::ParsedOpCompileInfoV2> parsed_op_compile_info_v2(                                   \
            new optiling::ParsedOpCompileInfoV2());                                                                   \
        parsed_op_compile_info_v2->value = cinfo_str;                                                                 \
        parsed_op_compile_info_v2->parsed_object = std::static_pointer_cast<void>(parsed_object_cinfo_v2);            \
        rwlock.wrlock();                                                                                              \
        parsed_compile_info_storage.emplace(hash_key.GetString(), parsed_op_compile_info_v2);                         \
        rwlock.unlock();                                                                                              \
    }                                                                                                                 \
    if (optiling::prof_switch) {                                                                                      \
        before_tiling = std::chrono::steady_clock::now();                                                             \
    }                                                                                                                 \
    bool ret_v2 = opfunc(op_type.GetString(), para, *parsed_object_cinfo_v2, rinfo);                                  \
    if (optiling::prof_switch) {                                                                                      \
        after_tiling = std::chrono::steady_clock::now();                                                              \
        uint64_t t0 = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - start_tiling).count();     \
        uint64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count();    \
        GEEVENT("[OPTILING_PROF] op_type: %s, total_us: %d, tiling_us: %d", op_type.GetString(), t0, t1);             \
    }                                                                                                                 \
    return ret_v2;                                                                                                    \
}                                                                                                                     \
REGISTER_OP_TILING_V2(optype, g_##optype##_TilingEntry_V2)

}
}  // namespace optiling
#endif // CANN_OPS_COMMON_OP_TILING_H_