#ifndef FUSED_MATRIX_COL_MAJOR_HPP
#define FUSED_MATRIX_COL_MAJOR_HPP

#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/fused_coord.hpp"
#include "../../../attn_infra/detail/fused_alignment.hpp"
#include "../../../attn_infra/fused_matrix_coord.hpp"

namespace NpuArch::layout {

/// Mapping function for col-major matrices
struct ColumnMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    HOST_DEVICE
    ColumnMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), LongIndex(rows))) {}

    /// Constructor
    HOST_DEVICE
    ColumnMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), ldm)) {}

    /// Ctor
    HOST_DEVICE
    ColumnMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
    }

    /// Returns the layout of a tile_common.
    HOST_DEVICE
    ColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return ColumnMajor(tileShape, stride());
    }

    /// Returns the shape of the layout
    HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    HOST_DEVICE
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    HOST_DEVICE
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    HOST_DEVICE
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    HOST_DEVICE
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

}  // namespace NpuArch::layout

#endif  // FUSED_MATRIX_COL_MAJOR_HPP