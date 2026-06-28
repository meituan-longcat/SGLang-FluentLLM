/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file lock.cpp
 * \brief
 */
#include "lock.h"

namespace optiling {
void RWLock::rdlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    _waiting_readers += 1;
    _read_cv.wait(lck, [&]() { return _waiting_writers == 0 && _status >= 0; });
    _waiting_readers -= 1;
    _status += 1;
}

void RWLock::wrlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    _waiting_writers += 1;
    _write_cv.wait(lck, [&]() { return _status == 0; });
    _waiting_writers -= 1;
    _status = -1;
}

void RWLock::unlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    if (_status == -1) {
        _status = 0;
    } else {
        _status -= 1;
    }
    if (_waiting_writers > 0) {
        if (_status == 0) {
            _write_cv.notify_one();
        }
    } else {
        _read_cv.notify_all();
    }
}
} // namespace optiling