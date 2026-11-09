#include "register/op_def_registry.h"

namespace ops {

class KvCacheCopyRows : public OpDef {
public:
    explicit KvCacheCopyRows(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> cacheTypes = {
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_BF16, ge::DT_INT8, ge::DT_INT8};
        const std::initializer_list<ge::DataType> indexTypes = {
            ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT32, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("cache")
            .ParamType(REQUIRED)
            .DataType(cacheTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Input("src_indices")
            .ParamType(REQUIRED)
            .DataType(indexTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Input("dst_indices")
            .ParamType(REQUIRED)
            .DataType(indexTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("cache")
            .ParamType(REQUIRED)
            .DataType(cacheTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(KvCacheCopyRows);

} // namespace ops
