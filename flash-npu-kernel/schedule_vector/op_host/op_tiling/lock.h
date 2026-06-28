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
 * \file lock.h
 * \brief temporary rwlock implementation from willzhang4a58.github.io
 */

#ifndef CANN_OPS_BUILT_IN_LOCK_H_
#define CANN_OPS_BUILT_IN_LOCK_H_

#include <mutex>
#include <condition_variable>

namespace optiling {
class RWLock {
public:
    RWLock() : _status(0), _waiting_readers(0), _waiting_writers(0) {}
    ~RWLock() {};
    RWLock(const RWLock&) = delete;
    RWLock(RWLock&&) = delete;
    RWLock& operator = (const RWLock&) = delete;
    RWLock& operator = (RWLock&&) = delete;

    void rdlock();

    void wrlock();

    void unlock();

private:
    // negative: writer
    // positive: reader
    int32_t _status;
    int32_t _waiting_readers;
    int32_t _waiting_writers;
    std::mutex _mtx;
    std::condition_variable _read_cv;
    std::condition_variable _write_cv;
};
} // namespace optiling
#endif // CANN_OPS_BUILT_IN_LOCK_H_