/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file fp16_t.cpp
 * \brief Half precision float
 */
#include "fp16_t.hpp"

namespace fe {
/**
 * @ingroup fp16_t global filed
 * @brief   round mode of last valid digital
 */
const fp16RoundMode_t g_RoundMode = ROUND_TO_NEAREST;

void ExtractFP16(const uint16_t& val, uint16_t* s, int16_t* e, uint16_t* m) {
  // 1.Extract
  *s = FP16_EXTRAC_SIGN(val);
  *e = FP16_EXTRAC_EXP(val);
  *m = FP16_EXTRAC_MAN(val);

  // Denormal
  if (0 == (*e)) {
    *e = 1;
  }
}
/**
 * @ingroup fp16_t static method
 * @param [in] man       truncated mantissa
 * @param [in] shiftOut left shift bits based on ten bits
 * @brief   judge whether to add one to the result while converting fp16_t to other datatype
 * @return  Return true if add one, otherwise false
 */
static bool IsRoundOne(uint64_t man, uint16_t truncLen) {
  uint16_t shiftOut = truncLen - DIM_2;
  uint64_t mask0 = 0x4;
  mask0 = mask0 << shiftOut;
  uint64_t mask1 = 0x2;
  mask1 = mask1 << shiftOut;
  uint64_t mask2;
  mask2 = mask1 - 1;

  bool lastBit = ((man & mask0) > 0);
  bool truncHigh = false;
  bool truncLeft = false;
  if (ROUND_TO_NEAREST == g_RoundMode) {
    truncHigh = ((man & mask1) > 0);
    truncLeft = ((man & mask2) > 0);
  }
  return (truncHigh && (truncLeft || lastBit));
}
/**
 * @ingroup fp16_t public method
 * @param [in] exp       exponent of fp16_t value
 * @param [in] man       exponent of fp16_t value
 * @brief   normalize fp16_t value
 * @return
 */
static void Fp16Normalize(int16_t& exp, uint16_t& man) {
  if (exp >= FP16_MAX_EXP) {
    exp = FP16_MAX_EXP - 1;
    man = FP16_MAX_MAN;
  } else if (exp == 0 && man == FP16_MAN_HIDE_BIT) {
    exp++;
    man = 0;
  }
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to float/fp32
 * @return  Return float/fp32 value of fpVal which is the value of fp16_t object
 */
static float fp16ToFloat(const uint16_t& fpVal) {
  float ret;

  uint16_t hfSign, hfMan;
  int16_t hfExp;
  ExtractFP16(fpVal, &hfSign, &hfExp, &hfMan);

  while (hfMan && !(hfMan & FP16_MAN_HIDE_BIT)) {
    hfMan <<= 1;
    hfExp--;
  }

  uint32_t sRet, eRet, mRet, fVal;

  sRet = hfSign;
  if (!hfMan) {
    eRet = 0;
    mRet = 0;
  } else {
    eRet = static_cast<uint32_t>(hfExp - FP16_EXP_BIAS + FP32_EXP_BIAS);
    mRet = hfMan & FP16_MAN_MASK;
    mRet = mRet << (FP32_MAN_LEN - FP16_MAN_LEN);
  }
  fVal = FP32_CONSTRUCTOR(sRet, eRet, mRet);
  uint32_t* ptrFVal = &fVal;
  float* ptrRet = reinterpret_cast<float*>(ptrFVal);
  ret = *ptrRet;

  return ret;
}
/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to double/fp64
 * @return  Return double/fp64 value of fpVal which is the value of fp16_t object
 */
static double fp16ToDouble(const uint16_t& fpVal) {
  double ret;

  uint16_t hfSign, hfMan;
  int16_t hfExp;
  ExtractFP16(fpVal, &hfSign, &hfExp, &hfMan);

  while (hfMan && !(hfMan & FP16_MAN_HIDE_BIT)) {
    hfMan <<= 1;
    hfExp--;
  }

  uint64_t sRet, eRet, mRet, fVal;

  sRet = hfSign;
  if (!hfMan) {
    eRet = 0;
    mRet = 0;
  } else {
    eRet = static_cast<uint64_t>(hfExp - FP16_EXP_BIAS + FP64_EXP_BIAS);
    mRet = hfMan & FP16_MAN_MASK;
    mRet = mRet << (FP64_MAN_LEN - FP16_MAN_LEN);
  }
  fVal = (sRet << FP64_SIGN_INDEX) | (eRet << FP64_MAN_LEN) | (mRet);
  uint64_t* ptrFVal = &fVal;
  double* ptrRet = reinterpret_cast<double*>(ptrFVal);
  ret = *ptrRet;

  return ret;
}
/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to int8_t
 * @return  Return int8_t value of fpVal which is the value of fp16_t object
 */
static int8_t fp16ToInt8(const uint16_t& fpVal) {
  int8_t ret;
  uint8_t retV;
  uint8_t sRet;
  uint8_t mRet = 0;
  uint16_t hfE, hfM;

  // 1.get sRet and shift it to bit0.
  sRet = FP16_EXTRAC_SIGN(fpVal);
  // 2.get hfE and hfM
  hfE = FP16_EXTRAC_EXP(fpVal);
  hfM = FP16_EXTRAC_MAN(fpVal);

  if (FP16_IS_DENORM(fpVal)) {  // Denormalized number
    retV = 0;
    ret = *((uint8_t*)&retV);
    return ret;
  }

  uint16_t shiftOut = 0;
  uint8_t overflowFlag = 0;
  uint64_t longIntM = hfM;

  if (FP16_IS_INVALID(fpVal)) {  // Inf or NaN
    overflowFlag = 1;
  } else {
    while (hfE != FP16_EXP_BIAS) {
      if (hfE > FP16_EXP_BIAS) {
        hfE--;
        longIntM = longIntM << 1;
        if (sRet == 1 && longIntM >= 0x20000u) {  // sign=1,negative number(<0)
          longIntM = 0x20000u;                    // 10 0000 0000 0000 0000  10(fp16_t-man)+7(int8)=17bit
          overflowFlag = 1;
          break;
        } else if (sRet != 1 && longIntM >= 0x1FFFFu) {  // sign=0,positive number(>0)
          longIntM = 0x1FFFFu;                           // 01 1111 1111 1111 1111  10(fp16_t-man)+7(int8)
          overflowFlag = 1;
          break;
        }
      } else {
        hfE++;
        shiftOut++;
      }
    }
  }
  if (overflowFlag) {
    retV = INT8_T_MAX + sRet;
  } else {
    bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
    mRet = (uint8_t)((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN8_MAX);
    needRound = needRound && ((sRet == 0 && mRet < INT8_T_MAX) || (sRet == 1 && mRet <= INT8_T_MAX));
    if (needRound) {
      mRet++;
    }
    if (sRet) {
      mRet = (~mRet) + DIM_1;
    }
    if (mRet == 0) {
      sRet = 0;
    }
    // Generate final result
    retV = (sRet << BitShift_7) | (mRet);
  }

  ret = *((uint8_t*)&retV);
  return ret;
}
/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to uint8_t
 * @return  Return uint8_t value of fpVal which is the value of fp16_t object
 */
static uint8_t fp16ToUInt8(const uint16_t& fpVal) {
  uint8_t ret;
  uint8_t sRet;
  uint8_t mRet = 0;
  uint16_t hfE, hfM;

  // 1.get sRet and shift it to bit0.
  sRet = FP16_EXTRAC_SIGN(fpVal);
  // 2.get hfE and hfM
  hfE = FP16_EXTRAC_EXP(fpVal);
  hfM = FP16_EXTRAC_MAN(fpVal);

  if (FP16_IS_DENORM(fpVal)) {  // Denormalized number
    return 0;
  }

  if (FP16_IS_INVALID(fpVal)) {  // Inf or NaN
    mRet = ~0;
  } else {
    uint16_t shiftOut = 0;
    uint64_t longIntM = hfM;
    uint8_t overflowFlag = 0;

    while (hfE != FP16_EXP_BIAS) {
      if (hfE > FP16_EXP_BIAS) {
        hfE--;
        longIntM = longIntM << 1;
        if (longIntM >= 0x40000Lu) {  // overflow 0100 0000 0000 0000 0000
          longIntM = 0x3FFFFLu;       // 11 1111 1111 1111 1111   10(fp16_t-man)+8(uint8)=18bit
          overflowFlag = 1;
          mRet = ~0;
          break;
        }
      } else {
        hfE++;
        shiftOut++;
      }
    }
    if (!overflowFlag) {
      bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
      mRet = (uint8_t)((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN8_MAX);
      if (needRound && mRet != BIT_LEN8_MAX) {
        mRet++;
      }
    }
  }

  if (sRet == 1) {  // Negative number
    mRet = 0;
  }
  // Generate final result
  ret = mRet;

  return ret;
}
/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to int16_t
 * @return  Return int16_t value of fpVal which is the value of fp16_t object
 */
static int16_t fp16ToInt16(const uint16_t& fpVal) {
  int16_t ret;
  uint16_t retV;
  uint16_t sRet;
  uint16_t mRet = 0;
  uint16_t hfE, hfM;

  // 1.get sRet and shift it to bit0.
  sRet = FP16_EXTRAC_SIGN(fpVal);
  // 2.get hfE and hfM
  hfE = FP16_EXTRAC_EXP(fpVal);
  hfM = FP16_EXTRAC_MAN(fpVal);

  if (FP16_IS_DENORM(fpVal)) {  // Denormalized number
    retV = 0;
    ret = *((uint8_t*)&retV);
    return ret;
  }

  uint16_t shiftOut = 0;
  uint64_t longIntM = hfM;
  uint8_t overflowFlag = 0;

  if (FP16_IS_INVALID(fpVal)) {  // Inf or NaN
    overflowFlag = 1;
  } else {
    while (hfE != FP16_EXP_BIAS) {
      if (hfE > FP16_EXP_BIAS) {
        hfE--;
        longIntM = longIntM << 1;
        if (sRet == 1 && longIntM > 0x2000000Lu) {  // sign=1,negative number(<0)
          longIntM = 0x2000000Lu;                   // 10(fp16_t-man)+15(int16)=25bit
          overflowFlag = 1;
          break;
        } else if (sRet != 1 && longIntM >= 0x1FFFFFFLu) {  // sign=0,positive number(>0) Overflow
          longIntM = 0x1FFFFFFLu;                           // 10(fp16_t-man)+15(int16)=25bit
          overflowFlag = 1;
          break;
        }
      } else {
        hfE++;
        shiftOut++;
      }
    }
  }
  if (overflowFlag) {
    retV = INT16_T_MAX + sRet;
  } else {
    bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
    mRet = (uint16_t)((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN16_MAX);
    if (needRound && mRet < INT16_T_MAX) {
      mRet++;
    }
    if (sRet) {
      mRet = (~mRet) + DIM_1;
    }
    if (mRet == 0) {
      sRet = 0;
    }
    // Generate final result
    retV = (sRet << BitShift_15) | (mRet);
  }

  ret = *((int16_t*)&retV);
  return ret;
}
/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to uint16_t
 * @return  Return uint16_t value of fpVal which is the value of fp16_t object
 */
static uint16_t fp16ToUInt16(const uint16_t& fpVal) {
  uint16_t ret;
  uint16_t sRet, mRet = 0;
  uint16_t hfE, hfM;

  // 1.get sRet and shift it to bit0.
  sRet = FP16_EXTRAC_SIGN(fpVal);
  // 2.get hfE and hfM
  hfE = FP16_EXTRAC_EXP(fpVal);
  hfM = FP16_EXTRAC_MAN(fpVal);

  if (FP16_IS_DENORM(fpVal)) {  // Denormalized number
    return 0;
  }

  if (FP16_IS_INVALID(fpVal)) {  // Inf or NaN
    mRet = ~0;
  } else {
    uint16_t shiftOut = 0;
    uint64_t longIntM = hfM;

    while (hfE != FP16_EXP_BIAS) {
      if (hfE > FP16_EXP_BIAS) {
        hfE--;
        longIntM = longIntM << 1;
      } else {
        hfE++;
        shiftOut++;
      }
    }
    bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
    mRet = (uint16_t)((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN16_MAX);
    if (needRound && mRet != BIT_LEN16_MAX) {
      mRet++;
    }
  }

  if (sRet == 1) {  // Negative number
    mRet = 0;
  }
  // Generate final result
  ret = mRet;

  return ret;
}
/**
 * @ingroup fp16_t math convertion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to int32_t
 * @return  Return int32_t value of fpVal which is the value of fp16_t object
 */
static int32_t fp16ToInt32(const uint16_t& fpVal) {
  int32_t ret;
  uint32_t retV;
  uint32_t sRet, mRet;
  uint16_t hfE, hfM;

  // 1.get sRet and shift it to bit0.
  sRet = FP16_EXTRAC_SIGN(fpVal);
  // 2.get hfE and hfM
  hfE = FP16_EXTRAC_EXP(fpVal);
  hfM = FP16_EXTRAC_MAN(fpVal);

  if (FP16_IS_INVALID(fpVal)) {  // Inf or NaN
    retV = INT32_T_MAX + sRet;
  } else {
    uint16_t shiftOut = 0;
    uint64_t longIntM = hfM;

    while (hfE != FP16_EXP_BIAS) {
      if (hfE > FP16_EXP_BIAS) {
        hfE--;
        longIntM = longIntM << 1;
      } else {
        hfE++;
        shiftOut++;
      }
    }
    bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
    mRet = (uint32_t)((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN32_MAX);
    if (needRound && mRet < INT32_T_MAX) {
      mRet++;
    }

    if (sRet == 1) {
      mRet = (~mRet) + 1;
    }
    if (mRet == 0) {
      sRet = 0;
    }
    // Generate final result
    retV = (sRet << BitShift_31) | (mRet);
  }

  ret = *((int32_t*)&retV);
  return ret;
}
/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to uint32_t
 * @return  Return uint32_t value of fpVal which is the value of fp16_t object
 */
static uint32_t fp16ToUInt32(const uint16_t& fpVal) {
  uint32_t ret;
  uint32_t sRet, mRet;
  uint16_t hfE, hfM;

  // 1.get sRet and shift it to bit0.
  sRet = FP16_EXTRAC_SIGN(fpVal);
  // 2.get hfE and hfM
  hfE = FP16_EXTRAC_EXP(fpVal);
  hfM = FP16_EXTRAC_MAN(fpVal);

  if (FP16_IS_DENORM(fpVal)) {  // Denormalized number
    return 0u;
  }

  if (FP16_IS_INVALID(fpVal)) {  // Inf or NaN
    mRet = ~0u;
  } else {
    uint16_t shiftOut = 0;
    uint64_t longIntM = hfM;

    while (hfE != FP16_EXP_BIAS) {
      if (hfE > FP16_EXP_BIAS) {
        hfE--;
        longIntM = longIntM << 1;
      } else {
        hfE++;
        shiftOut++;
      }
    }
    bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
    mRet = (uint32_t)(longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN32_MAX;
    if (needRound && mRet != BIT_LEN32_MAX) {
      mRet++;
    }
  }

  if (sRet == 1) {  // Negative number
    mRet = 0;
  }
  // Generate final result
  ret = mRet;

  return ret;
}
/**
 * @ingroup fp16_t math operator
 * @param [in] v1 left operator value of fp16_t object
 * @param [in] v2 right operator value of fp16_t object
 * @brief   Performing fp16_t addition
 * @return  Return fp16_t result of adding this and fp
 */
static uint16_t fp16Add(uint16_t v1, uint16_t v2) {
  uint16_t ret;
  uint16_t sA, sB;
  int16_t eA, eB;
  uint32_t mA, mB;
  uint16_t sRet, mRet;
  int16_t eRet;
  uint16_t maTmp, mbTmp;

  uint16_t shiftOut = 0;
  // 1.Extract
  ExtractFP16(v1, &sA, &eA, &maTmp);
  ExtractFP16(v2, &sB, &eB, &mbTmp);
  mA = maTmp;
  mB = mbTmp;

  uint16_t sum;
  if (sA != sB) {
    ReverseMan(sA > 0, &mA);
    ReverseMan(sB > 0, &mB);
    sum = (uint16_t)GetManSum(eA, mA, eB, mB);
    sRet = (sum & FP16_SIGN_MASK) >> FP16_SIGN_INDEX;
    ReverseMan(sRet > 0, &mA);
    ReverseMan(sRet > 0, &mB);
  } else {
    sum = (uint16_t)GetManSum(eA, mA, eB, mB);
    sRet = sA;
  }

  if (sum == 0) {
    shiftOut = 3;
    mA = mA << shiftOut;
    mB = mB << shiftOut;
  }

  int16_t eTmp = 0;
  uint32_t mTrunc = 0;

  eRet = std::max(eA, eB);
  eTmp = std::abs(eA - eB);
  if (eA > eB) {
    mTrunc = (mB << (32 - (uint16_t)eTmp));
    mB = RightShift(mB, eTmp);
  } else if (eA < eB) {
    mTrunc = (mA << (32 - (uint16_t)eTmp));
    mA = RightShift(mA, eTmp);
  }
  // calculate mantissa
  mRet = (uint16_t)(mA + mB);

  uint16_t m_min = static_cast<uint16_t>(FP16_MAN_HIDE_BIT << shiftOut);
  uint16_t m_max = m_min << 1;
  // Denormal
  while (mRet < m_min && eRet > 0) {  // the value of mRet should not be smaller than 2^23
    mRet = mRet << 1;
    mRet += (FP32_SIGN_MASK & mTrunc) >> FP32_SIGN_INDEX;
    mTrunc = mTrunc << 1;
    eRet = eRet - 1;
  }
  while (mRet >= m_max) {  // the value of mRet should be smaller than 2^24
    mTrunc = mTrunc >> 1;
    mTrunc = mTrunc | (FP32_SIGN_MASK * (mRet & 1));
    mRet = mRet >> 1;
    eRet = eRet + 1;
  }

  bool b_last_bit = ((mRet & 1) > 0);
  bool b_trunc_high = 0;
  bool b_trunc_left = 0;
  b_trunc_high = (ROUND_TO_NEAREST == g_RoundMode) && ((mTrunc & FP32_SIGN_MASK) > 0);
  b_trunc_left = (ROUND_TO_NEAREST == g_RoundMode) && ((mTrunc & FP32_ABS_MAX) > 0);
  mRet = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, mRet, shiftOut);
  while (mRet >= m_max) {
    mRet = mRet >> 1;
    eRet = eRet + 1;
  }

  if (eRet == 0 && mRet <= m_max) {
    mRet = mRet >> 1;
  }
  Fp16Normalize(eRet, mRet);

  ret = FP16_CONSTRUCTOR(sRet, (uint16_t)eRet, mRet);
  return ret;
}
/**
 * @ingroup fp16_t math operator
 * @param [in] v1 left operator value of fp16_t object
 * @param [in] v2 right operator value of fp16_t object
 * @brief   Performing fp16_t subtraction
 * @return  Return fp16_t result of subtraction fp from this
 */
static uint16_t fp16Sub(uint16_t v1, uint16_t v2) {
  uint16_t ret;
  uint16_t tmp;

  // Reverse
  tmp = ((~(v2)) & FP16_SIGN_MASK) | (v2 & FP16_ABS_MAX);

  ret = fp16Add(v1, tmp);

  return ret;
}
/**
 * @ingroup fp16_t math operator
 * @param [in] v1 left operator value of fp16_t object
 * @param [in] v2 right operator value of fp16_t object
 * @brief   Performing fp16_t multiplication
 * @return  Return fp16_t result of multiplying this and fp
 */
static uint16_t fp16Mul(uint16_t v1, uint16_t v2) {
  uint16_t sA, sB;
  int16_t eA, eB;
  uint32_t mA, mB;
  uint16_t sRet, mRet;
  int16_t eRet;
  uint32_t mulM;
  uint16_t maTmp, mbTmp;
  // 1.Extract
  ExtractFP16(v1, &sA, &eA, &maTmp);
  ExtractFP16(v2, &sB, &eB, &mbTmp);
  mA = maTmp;
  mB = mbTmp;

  eRet = eA + eB - FP16_EXP_BIAS - DIM_10;
  mulM = mA * mB;
  sRet = sA ^ sB;

  uint32_t mMin = FP16_MAN_HIDE_BIT;
  uint32_t mMax = mMin << 1;
  uint32_t mTrunc = 0;
  // the value of mRet should not be smaller than 2^23
  while (mulM < mMin && eRet > 1) {
    mulM = mulM << 1;
    eRet = eRet - 1;
  }
  while (mulM >= mMax || eRet < 1) {
    mTrunc = mTrunc >> 1;
    mTrunc = mTrunc | (FP32_SIGN_MASK * (mulM & 1));
    mulM = mulM >> 1;
    eRet = eRet + 1;
  }
  bool bLastBit = ((mulM & 1) > 0);
  bool bTruncHigh = (ROUND_TO_NEAREST == g_RoundMode) && ((mTrunc & FP32_SIGN_MASK) > 0);
  bool bTruncLeft = (ROUND_TO_NEAREST == g_RoundMode) && ((mTrunc & FP32_ABS_MAX) > 0);
  mulM = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mulM);

  while (mulM >= mMax || eRet < 0) {
    mulM = mulM >> 1;
    eRet = eRet + 1;
  }

  if (eRet == 1 && mulM < FP16_MAN_HIDE_BIT) {
    eRet = 0;
  }
  mRet = (uint16_t)mulM;

  Fp16Normalize(eRet, mRet);

  uint16_t ret = FP16_CONSTRUCTOR(sRet, (uint16_t)eRet, mRet);
  return ret;
}
/**
 * @ingroup fp16_t math operator divided
 * @param [in] v1 left operator value of fp16_t object
 * @param [in] v2 right operator value of fp16_t object
 * @brief   Performing fp16_t division
 * @return  Return fp16_t result of division this by fp
 */
static uint16_t fp16Div(uint16_t v1, uint16_t v2) {
  uint16_t ret;
  if (FP16_IS_ZERO(v2)) {  // result is inf
    // throw "fp16_t division by zero.";
    uint16_t sA, sB;
    uint16_t sRet;
    sA = FP16_EXTRAC_SIGN(v1);
    sB = FP16_EXTRAC_SIGN(v2);
    sRet = sA ^ sB;
    ret = FP16_CONSTRUCTOR(sRet, FP16_MAX_EXP - 1, FP16_MAX_MAN);
  } else if (FP16_IS_ZERO(v1)) {
    ret = 0u;
  } else {
    uint16_t sA, sB;
    int16_t eA, eB;
    uint64_t mA, mB;
    float mDiv;
    uint16_t maTmp, mbTmp;
    // 1.Extract
    ExtractFP16(v1, &sA, &eA, &maTmp);
    ExtractFP16(v2, &sB, &eB, &mbTmp);
    mA = maTmp;
    mB = mbTmp;

    uint64_t mTmp;
    if (eA > eB) {
      mTmp = mA;
      uint16_t tmp;
      tmp = static_cast<uint16_t>(eA - eB);
      for (int i = 0; i < tmp; i++) {
        mTmp = mTmp << 1;
      }
      mA = mTmp;
    } else if (eA < eB) {
      mTmp = mB;
      uint16_t tmp = static_cast<uint16_t>(eB - eA);
      for (int i = 0; i < tmp; i++) {
        mTmp = mTmp << 1;
      }
      mB = mTmp;
    }
    mDiv = (float)(mA * 1.0f / mB);
    fp16_t fpDiv = mDiv; /*lint !e524*/
    ret = fpDiv.val;
    if (sA != sB) {
      ret |= FP16_SIGN_MASK;
    }
  }
  return ret;
}

// operate
fp16_t fp16_t::operator+(const fp16_t fp) {
  uint16_t retVal = fp16Add(val, fp.val);
  fp16_t ret(retVal);
  return ret;
}
fp16_t fp16_t::operator-(const fp16_t fp) {
  uint16_t retVal = fp16Sub(val, fp.val);
  fp16_t ret(retVal);
  return ret;
}
fp16_t fp16_t::operator*(const fp16_t fp) {
  uint16_t retVal = fp16Mul(val, fp.val);
  fp16_t ret(retVal);
  return ret;
}
fp16_t fp16_t::operator/(const fp16_t fp) {
  uint16_t retVal = fp16Div(val, fp.val);
  fp16_t ret(retVal);
  return ret;
}

fp16_t fp16_t::operator+=(const fp16_t fp) {
  val = fp16Add(val, fp.val);
  return *this;
}
fp16_t fp16_t::operator-=(const fp16_t fp) {
  val = fp16Sub(val, fp.val);
  return *this;
}
fp16_t fp16_t::operator*=(const fp16_t fp) {
  val = fp16Mul(val, fp.val);
  return *this;
}
fp16_t fp16_t::operator/=(const fp16_t fp) {
  val = fp16Div(val, fp.val);
  return *this;
}

// compare
bool fp16_t::operator==(const fp16_t& fp) const {
  bool result = true;
  if (FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val)) {
    result = true;
  } else {
    result = ((val & BIT_LEN16_MAX) == (fp.val & BIT_LEN16_MAX));  // bit compare
  }
  return result;
}
bool fp16_t::operator!=(const fp16_t& fp) const {
  bool result = true;
  if (FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val)) {
    result = false;
  } else {
    result = ((val & BIT_LEN16_MAX) != (fp.val & BIT_LEN16_MAX));  // bit compare
  }
  return result;
}
bool fp16_t::operator>(const fp16_t& fp) const {
  uint16_t sA, sB;
  uint16_t eA, eB;
  uint16_t mA, mB;
  bool result = true;

  // 1.Extract
  sA = FP16_EXTRAC_SIGN(val);
  sB = FP16_EXTRAC_SIGN(fp.val);
  eA = FP16_EXTRAC_EXP(val);
  eB = FP16_EXTRAC_EXP(fp.val);
  mA = FP16_EXTRAC_MAN(val);
  mB = FP16_EXTRAC_MAN(fp.val);

  // Compare
  if ((sA == 0) && (sB > 0)) {  // +  -
    // -0=0
    result = !(FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val));
  } else if ((sA == 0) && (sB == 0)) {  // + +
    if (eA > eB) {                      // eA - eB >= 1; Va always larger than Vb
      result = true;
    } else if (eA == eB) {
      result = mA > mB;
    } else {
      result = false;
    }
  } else if ((sA > 0) && (sB > 0)) {  // - -    opposite to  + +
    if (eA < eB) {
      result = true;
    } else if (eA == eB) {
      result = mA < mB;
    } else {
      result = false;
    }
  } else {  // -  +
    result = false;
  }

  return result;
}
bool fp16_t::operator>=(const fp16_t& fp) const {
  bool result = true;
  if ((*this) > fp) {
    result = true;
  } else if ((*this) == fp) {
    result = true;
  } else {
    result = false;
  }

  return result;
}
bool fp16_t::operator<(const fp16_t& fp) const {
  bool result = true;
  if ((*this) >= fp) {
    result = false;
  } else {
    result = true;
  }

  return result;
}
bool fp16_t::operator<=(const fp16_t& fp) const {
  bool result = true;
  if ((*this) > fp) {
    result = false;
  } else {
    result = true;
  }

  return result;
}

// evaluation
fp16_t& fp16_t::operator=(const fp16_t& fp) {
  if (this == &fp) {
    return *this;
  }
  val = fp.val;
  return *this;
}
fp16_t& fp16_t::operator=(const float& fVal) {
  uint16_t sRet, mRet;
  int16_t eRet;
  uint32_t eF, mF;
  uint32_t ui32V = *(reinterpret_cast<const uint32_t*>(&fVal));  // 1:8:23bit sign:exp:man
  uint32_t mLenDelta;

  sRet = static_cast<uint16_t>((ui32V & FP32_SIGN_MASK) >> FP32_SIGN_INDEX);  // 4Byte->2Byte
  eF = (ui32V & FP32_EXP_MASK) >> FP32_MAN_LEN;                    // 8 bit exponent
  mF = (ui32V & FP32_MAN_MASK);                                    // 23 bit mantissa dont't need to care about denormal
  mLenDelta = FP32_MAN_LEN - FP16_MAN_LEN;

  bool needRound = false;
  // Exponent overflow/NaN converts to signed inf/NaN
  if (eF > 0x8Fu) {  // 0x8Fu:142=127+15
    eRet = FP16_MAX_EXP - 1;
    mRet = FP16_MAX_MAN;
  } else if (eF <= 0x70u) {  // 0x70u:112=127-15 Exponent underflow converts to denormalized half or signed zero
    eRet = 0;
    if (eF >= 0x67) {  // 0x67:103=127-24 Denormal
      mF = (mF | FP32_MAN_HIDE_BIT);
      uint16_t shiftOut = FP32_MAN_LEN;
      uint64_t mTmp = ((uint64_t)mF) << (eF - 0x67);

      needRound = IsRoundOne(mTmp, shiftOut);
      mRet = static_cast<uint16_t>(mTmp >> shiftOut);
      if (needRound) {
        mRet++;
      }
    } else if (eF == 0x66 && mF > 0) {  // 0x66:102 Denormal 0<f_v<min(Denormal)
      mRet = 1;
    } else {
      mRet = 0;
    }
  } else {  // Regular case with no overflow or underflow
    eRet = (int16_t)(eF - 0x70u);

    needRound = IsRoundOne(mF, mLenDelta);
    mRet = static_cast<uint16_t>(mF >> mLenDelta);
    if (needRound) {
      mRet++;
    }
    if (mRet & FP16_MAN_HIDE_BIT) {
      eRet++;
    }
  }

  Fp16Normalize(eRet, mRet);
  val = FP16_CONSTRUCTOR(sRet, (uint16_t)eRet, mRet);
  return *this;
}
fp16_t& fp16_t::operator=(const int8_t& iVal) {
  uint16_t sRet, eRet, mRet;

  sRet = (uint16_t)((((uint8_t)iVal) & 0x80) >> DIM_7);
  mRet = (uint16_t)((((uint8_t)iVal) & INT8_T_MAX));

  if (mRet == 0) {
    eRet = 0;
  } else {
    if (sRet) {                         // negative number(<0)
      mRet = (uint16_t)std::abs(iVal);  // complement
    }

    eRet = FP16_MAN_LEN;
    while ((mRet & FP16_MAN_HIDE_BIT) == 0) {
      mRet = mRet << DIM_1;
      eRet = eRet - DIM_1;
    }
    eRet = eRet + FP16_EXP_BIAS;
  }

  val = FP16_CONSTRUCTOR(sRet, eRet, mRet);
  return *this;
}
fp16_t& fp16_t::operator=(const uint8_t& uiVal) {
  uint16_t sRet, eRet, mRet;
  sRet = 0;
  eRet = 0;
  mRet = uiVal;
  if (mRet) {
    eRet = FP16_MAN_LEN;
    while ((mRet & FP16_MAN_HIDE_BIT) == 0) {
      mRet = mRet << DIM_1;
      eRet = eRet - DIM_1;
    }
    eRet = eRet + FP16_EXP_BIAS;
  }

  val = FP16_CONSTRUCTOR(sRet, eRet, mRet);
  return *this;
}
fp16_t& fp16_t::operator=(const int16_t& iVal) {
  if (iVal == 0) {
    val = 0;
  } else {
    uint16_t sRet;
    uint16_t uiVal = *(reinterpret_cast<const uint16_t*>(&iVal));
    sRet = (uint16_t)(uiVal >> BitShift_15);
    if (sRet) {
      int16_t iValM = -iVal;
      uiVal = *(reinterpret_cast<uint16_t*>(&iValM));
    }
    uint32_t mTmp = (uiVal & FP32_ABS_MAX);

    uint16_t mMin = FP16_MAN_HIDE_BIT;
    uint16_t mMax = mMin << 1;
    uint16_t len = (uint16_t)GetManBitLength(mTmp);
    if (mTmp) {
      int16_t eRet;

      if (len > DIM_11) {
        eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
        uint16_t eTmp = len - DIM_11;
        uint32_t truncMask = 1;
        for (int i = 1; i < eTmp; i++) {
          truncMask = (truncMask << 1) + 1;
        }
        uint32_t mTrunc = (mTmp & truncMask) << (BitShift_32 - eTmp);
        for (int i = 0; i < eTmp; i++) {
          mTmp = (mTmp >> 1);
          eRet = eRet + 1;
        }
        bool bLastBit = ((mTmp & 1) > 0);
        bool bTruncHigh = false;
        bool bTruncLeft = false;
        if (ROUND_TO_NEAREST == g_RoundMode) {  // trunc
          bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
          bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
        }
        mTmp = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mTmp);
        while (mTmp >= mMax || eRet < 0) {
          mTmp = mTmp >> 1;
          eRet = eRet + 1;
        }
      } else {
        eRet = FP16_EXP_BIAS;
        mTmp = mTmp << (11 - len);
        eRet = eRet + (len - 1);
      }
      uint16_t mRet = (uint16_t)mTmp;
      val = FP16_CONSTRUCTOR(sRet, (uint16_t)eRet, mRet);
    } else {
    }
  }
  return *this;
}
fp16_t& fp16_t::operator=(const uint16_t& uiVal) {
  if (uiVal == 0) {
    val = 0;
  } else {
    int16_t eRet;
    uint16_t mRet = uiVal;

    uint16_t mMin = FP16_MAN_HIDE_BIT;
    uint16_t mMax = mMin << 1;
    uint16_t len = (uint16_t)GetManBitLength(mRet);

    if (len > 11) {
      eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
      uint32_t mTrunc;
      uint32_t truncMask = 1;
      uint16_t eTmp = len - DIM_11;
      for (int i = 1; i < eTmp; i++) {
        truncMask = (truncMask << 1) + 1;
      }
      mTrunc = (mRet & truncMask) << (32 - eTmp);
      for (int i = 0; i < eTmp; i++) {
        mRet = (mRet >> 1);
        eRet = eRet + 1;
      }
      bool bLastBit = ((mRet & 1) > 0);
      bool bTruncHigh = false;
      bool bTruncLeft = false;
      if (ROUND_TO_NEAREST == g_RoundMode) {  // trunc
        bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
        bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
      }
      mRet = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mRet);
      while (mRet >= mMax || eRet < 0) {
        mRet = mRet >> 1;
        eRet = eRet + 1;
      }
      if (FP16_IS_INVALID(val)) {
        val = FP16_MAX;
      }
    } else {
      eRet = FP16_EXP_BIAS;
      mRet = mRet << (DIM_11 - len);
      eRet = eRet + (len - 1);
    }
    val = FP16_CONSTRUCTOR(0u, (uint16_t)eRet, mRet);
  }
  return *this;
}
fp16_t& fp16_t::operator=(const int32_t& iVal) {
  if (iVal == 0) {
    val = 0;
  } else {
    uint32_t uiVal = *(reinterpret_cast<const uint32_t*>(&iVal));
    uint16_t sRet = (uint16_t)(uiVal >> BitShift_31);
    if (sRet) {
      int32_t iValM = -iVal;
      uiVal = *(reinterpret_cast<uint32_t*>(&iValM));
    }
    int16_t eRet;
    uint32_t mTmp = (uiVal & FP32_ABS_MAX);

    uint32_t mMin = FP16_MAN_HIDE_BIT;
    uint32_t mMax = mMin << 1;
    uint16_t len = (uint16_t)GetManBitLength(mTmp);

    if (len > DIM_11) {
      eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
      uint32_t mTrunc = 0;
      uint32_t truncMask = 1;
      uint16_t eTmp = len - DIM_11;
      for (int i = 1; i < eTmp; i++) {
        truncMask = (truncMask << 1) + 1;
      }
      mTrunc = (mTmp & truncMask) << (BitShift_32 - eTmp);
      for (int i = 0; i < eTmp; i++) {
        mTmp = (mTmp >> 1);
        eRet = eRet + 1;
      }
      bool bLastBit = ((mTmp & 1) > 0);
      bool bTruncHigh = false;
      bool bTruncLeft = false;
      if (ROUND_TO_NEAREST == g_RoundMode) {  // trunc
        bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
        bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
      }
      mTmp = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mTmp);
      while (mTmp >= mMax || eRet < 0) {
        mTmp = mTmp >> 1;
        eRet = eRet + 1;
      }
      if (eRet >= FP16_MAX_EXP) {
        eRet = FP16_MAX_EXP - 1;
        mTmp = FP16_MAX_MAN;
      }
    } else {
      eRet = FP16_EXP_BIAS;
      mTmp = mTmp << (DIM_11 - len);
      eRet = eRet + (len - 1);
    }
    uint16_t mRet = (uint16_t)mTmp;
    val = FP16_CONSTRUCTOR(sRet, (uint16_t)eRet, mRet);
  }
  return *this;
}
fp16_t& fp16_t::operator=(const uint32_t& uiVal) {
  if (uiVal == 0) {
    val = 0;
  } else {
    int16_t eRet;
    uint32_t mTmp = uiVal;

    uint32_t mMin = FP16_MAN_HIDE_BIT;
    uint32_t mMax = mMin << 1;
    uint16_t len = (uint16_t)GetManBitLength(mTmp);

    if (len > DIM_11) {
      eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
      uint32_t mTrunc = 0;
      uint32_t truncMask = 1;
      uint16_t eTmp = len - DIM_11;
      for (int i = 1; i < eTmp; i++) {
        truncMask = (truncMask << 1) + 1;
      }
      mTrunc = (mTmp & truncMask) << (BitShift_32 - eTmp);
      for (int i = 0; i < eTmp; i++) {
        mTmp = (mTmp >> 1);
        eRet = eRet + 1;
      }
      bool bLastBit = ((mTmp & 1) > 0);
      bool bTruncHigh = false;
      bool bTruncLeft = false;
      if (ROUND_TO_NEAREST == g_RoundMode) {  // trunc
        bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
        bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
      }
      mTmp = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mTmp);
      while (mTmp >= mMax || eRet < 0) {
        mTmp = mTmp >> 1;
        eRet = eRet + 1;
      }
      if (eRet >= FP16_MAX_EXP) {
        eRet = FP16_MAX_EXP - 1;
        mTmp = FP16_MAX_MAN;
      }
    } else {
      eRet = FP16_EXP_BIAS;
      mTmp = mTmp << (DIM_11 - len);
      eRet = eRet + (len - 1);
    }
    uint16_t mRet = (uint16_t)mTmp;
    val = FP16_CONSTRUCTOR(0u, (uint16_t)eRet, mRet);
  }
  return *this;
}
fp16_t& fp16_t::operator=(const double& dVal) {
  uint16_t sRet, mRet;
  int16_t eRet;
  uint64_t eD, mD;
  uint64_t ui64V = *((uint64_t*)&dVal);  // 1:11:52bit sign:exp:man
  uint32_t mLenDelta;

  sRet = static_cast<uint16_t>((ui64V & FP64_SIGN_MASK) >> FP64_SIGN_INDEX);  // 4Byte
  eD = (ui64V & FP64_EXP_MASK) >> FP64_MAN_LEN;                    // 10 bit exponent
  mD = (ui64V & FP64_MAN_MASK);                                    // 52 bit mantissa
  mLenDelta = FP64_MAN_LEN - FP16_MAN_LEN;

  bool needRound = false;
  // Exponent overflow/NaN converts to signed inf/NaN
  if (eD >= 0x410u) {  // 0x410:1040=1023+16
    eRet = FP16_MAX_EXP - 1;
    mRet = FP16_MAX_MAN;
    val = FP16_CONSTRUCTOR(sRet, (uint16_t)eRet, mRet);
  } else if (eD <= 0x3F0u) {  // Exponent underflow converts to denormalized half or signed zero
    // 0x3F0:1008=1023-15
    /**
     * Signed zeros, denormalized floats, and floats with small
     * exponents all convert to signed zero half precision.
     */
    eRet = 0;
    if (eD >= 0x3E7u) {  // 0x3E7u:999=1023-24 Denormal
      // Underflows to a denormalized value
      mD = (FP64_MAN_HIDE_BIT | mD);
      uint16_t shiftOut = FP64_MAN_LEN;
      uint64_t mTmp = ((uint64_t)mD) << (eD - 0x3E7u);

      needRound = IsRoundOne(mTmp, shiftOut);
      mRet = static_cast<uint16_t>(mTmp >> shiftOut);
      if (needRound) {
        mRet++;
      }
    } else if (eD == 0x3E6u && mD > 0) {
      mRet = 1;
    } else {
      mRet = 0;
    }
  } else {  // Regular case with no overflow or underflow
    eRet = (int16_t)(eD - 0x3F0u);

    needRound = IsRoundOne(mD, mLenDelta);
    mRet = static_cast<uint16_t>(mD >> mLenDelta);
    if (needRound) {
      mRet++;
    }
    if (mRet & FP16_MAN_HIDE_BIT) {
      eRet++;
    }
  }

  Fp16Normalize(eRet, mRet);
  val = FP16_CONSTRUCTOR(sRet, (uint16_t)eRet, mRet);
  return *this;
}

// convert
fp16_t::operator float() const {
  return fp16ToFloat(val);
}
fp16_t::operator double() const {
  return fp16ToDouble(val);
}
fp16_t::operator int8_t() const {
  return fp16ToInt8(val);
}
fp16_t::operator uint8_t() const {
  return fp16ToUInt8(val);
}
fp16_t::operator int16_t() const {
  return fp16ToInt16(val);
}
fp16_t::operator uint16_t() const {
  return fp16ToUInt16(val);
}
fp16_t::operator int32_t() const {
  return fp16ToInt32(val);
}
fp16_t::operator uint32_t() const {
  return fp16ToUInt32(val);
}
int fp16_t::IsInf() {
  if ((val & FP16_ABS_MAX) == FP16_EXP_MASK) {
    if (val & FP16_SIGN_MASK) {
      return -1;
    } else {
      return 1;
    }
  } else {
    return 0;
  }
}

float fp16_t::toFloat() {
  return fp16ToFloat(val);
}
double fp16_t::toDouble() {
  return fp16ToDouble(val);
}
int8_t fp16_t::toInt8() {
  return fp16ToInt8(val);
}
uint8_t fp16_t::toUInt8() {
  return fp16ToUInt8(val);
}
int16_t fp16_t::toInt16() {
  return fp16ToInt16(val);
}
uint16_t fp16_t::toUInt16() {
  return fp16ToUInt16(val);
}
int32_t fp16_t::toInt32() {
  return fp16ToInt32(val);
}
uint32_t fp16_t::toUInt32() {
  return fp16ToUInt32(val);
}
}  // namespace fe
