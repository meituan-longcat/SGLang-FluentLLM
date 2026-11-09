#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_SUBCORE_INC_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_SUBCORE_INC_HPP

    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput, const LayoutOutput &layoutOutput,
        uint32_t rowOffset, uint32_t isFirstStackTile, uint32_t isFirstRowLoop,
        uint32_t columnNumRound, uint32_t pingpongFlag,
        uint32_t curStackTileMod, bool SplitKVFlag)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = NpuArch::Detail::Alignment::RoundUp(rowNumCurLoop, BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseMode::OUT_ONLY) {
            // In lse out-only mode, tv is used in the last stack tile to transport lse
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        } else {
            if (isFirstStackTile && isFirstRowLoop && SplitKVFlag) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        UpdateGlobalRowMax(rowNumCurLoop,
            rowNumCurLoopRound,
            columnNum,
            columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile);
        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        MoveP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        UpdateGlobalRowSum(
            sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
    }

#endif // FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_SUBCORE_INC_HPP
