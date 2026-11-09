// hostfunc/_binding.cpp
//
// pybind11 binding that launches a Python host function onto a CUDA stream via
// cudaLaunchHostFunc, in a way that can be *captured into* and *replayed by* a
// CUDA graph.
//
// This mirrors TensorRT-LLM's cpp/tensorrt_llm/nanobind/runtime/hostfunc.cpp,
// but uses pybind11 instead of nanobind so it can be JIT-compiled with
// torch.utils.cpp_extension.load (no extra build system required).
//
// A CUDA graph can only capture GPU kernels; pure-CPU work (e.g. grammar
// advance / mask generation) cannot be captured directly. cudaLaunchHostFunc
// enqueues a CPU callback onto a stream so it (a) is ordered against GPU work
// via stream semantics / events, and (b) is captured by a CUDA graph and
// replayed every time the graph replays.
//
// Lifetime is the crux of capture/replay correctness, governed by freeUserData:
//   * freeUserData == true  (NOT capturing; eager / prefill): the trampoline
//     frees the user data right after running -> no leak.
//   * freeUserData == false (capturing into a graph): the trampoline does NOT
//     free it, so the same callback + args survive and can be replayed N times.
//     The caller must free it later via free_hostfunc_user_data (or at exit).

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

// Mark hidden to match pybind11's -fvisibility=hidden on the module and avoid
// "declared with greater visibility than the type of its field" warnings.
struct __attribute__((visibility("hidden"))) HostFuncUserData {
    bool freeUserData;
    py::function pyHostFunc;
    py::tuple pyArgs;
    py::dict pyKwargs;

    HostFuncUserData(bool freeUserData, py::function func, py::tuple args, py::dict kwargs)
        : freeUserData(freeUserData)
        , pyHostFunc(std::move(func))
        , pyArgs(std::move(args))
        , pyKwargs(std::move(kwargs))
    {
    }
};

// Runs on a CUDA internal thread (NOT a Python thread), so we must acquire the
// GIL before touching any Python object.
void CUDART_CB cudaHostFuncTrampoline(void* userData)
{
    py::gil_scoped_acquire gil;

    // Take ownership; the unique_ptr destructor frees it unless we release().
    auto hostFuncUserData = std::unique_ptr<HostFuncUserData>(static_cast<HostFuncUserData*>(userData));
    try
    {
        // Equivalent to: pyHostFunc(*pyArgs, **pyKwargs)
        PyObject* ret = PyObject_Call(
            hostFuncUserData->pyHostFunc.ptr(),
            hostFuncUserData->pyArgs.ptr(),
            hostFuncUserData->pyKwargs.size() ? hostFuncUserData->pyKwargs.ptr() : nullptr);
        if (ret == nullptr)
        {
            // We cannot propagate a Python exception across the C ABI boundary;
            // print it so it is not silently swallowed.
            PyErr_Print();
        }
        else
        {
            Py_DECREF(ret);
        }
    }
    catch (py::error_already_set& e)
    {
        e.restore();
        PyErr_Print();
    }

    if (!hostFuncUserData->freeUserData)
    {
        // Capturing: keep the user data alive so the graph can replay it.
        hostFuncUserData.release();
    }
}

// Returns the user-data pointer (as uintptr_t) when freeUserData is false so the
// caller can free it later; returns std::nullopt when freeUserData is true.
std::optional<std::uintptr_t> launch_hostfunc(
    std::uintptr_t streamPtr, bool freeUserData, py::function pyHostFunc, py::args pyArgs, py::kwargs pyKwargs)
{
    auto const stream = reinterpret_cast<cudaStream_t>(streamPtr);

    auto hostFuncUserData = std::make_unique<HostFuncUserData>(
        freeUserData, std::move(pyHostFunc), py::tuple(pyArgs), py::dict(pyKwargs));

    cudaError_t err = cudaLaunchHostFunc(stream, cudaHostFuncTrampoline, hostFuncUserData.get());
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("cudaLaunchHostFunc failed: ") + cudaGetErrorString(err));
    }

    // Release C++ ownership; lifetime is now governed by the trampoline /
    // free_hostfunc_user_data depending on freeUserData.
    auto userDataPtr = reinterpret_cast<std::uintptr_t>(hostFuncUserData.release());
    return freeUserData ? std::nullopt : std::make_optional(userDataPtr);
}

void free_hostfunc_user_data(std::uintptr_t userDataPtr)
{
    // Re-take ownership and let the unique_ptr destructor free it.
    auto hostFuncUserData = std::unique_ptr<HostFuncUserData>(
        reinterpret_cast<HostFuncUserData*>(userDataPtr));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("launch_hostfunc", &launch_hostfunc, "Launch a Python host function onto a CUDA stream");
    m.def("free_hostfunc_user_data", &free_hostfunc_user_data, "Free host function user data");
}
