/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file fp16_t.hpp
 * \brief Half precision float
 */
#ifndef _FP16_T_HPP_
#define _FP16_T_HPP_

#include <stdint.h>
#include <math.h>
#include <cmath>
#include <algorithm>

namespace fe {

const uint16_t DIM_0 = 0;
const uint16_t DIM_1 = 1;
const uint16_t DIM_2 = 2;
const uint16_t DIM_3 = 3;
const uint16_t DIM_4 = 4;
const uint16_t DIM_5 = 5;
const uint16_t DIM_6 = 6;
const uint16_t DIM_7 = 7;
const uint16_t DIM_8 = 8;
const uint16_t DIM_9 = 9;
const uint16_t DIM_10 = 10;
const uint16_t DIM_11 = 11;
const uint16_t DIM_12 = 12;
const uint16_t DIM_13 = 13;
const uint16_t DIM_14 = 14;
const uint16_t DIM_15 = 15;
const uint16_t DIM_16 = 16;

const uint16_t BitShift_2 = 2;
const uint16_t BitShift_3 = 3;
const uint16_t BitShift_4 = 4;
const uint16_t BitShift_5 = 5;
const uint16_t BitShift_6 = 6;
const uint16_t BitShift_7 = 7;
const uint16_t BitShift_8 = 8;
const uint16_t BitShift_9 = 9;
const uint16_t BitShift_10 = 10;
const uint16_t BitShift_12 = 12;
const uint16_t BitShift_13 = 13;
const uint16_t BitShift_14 = 14;
const uint16_t BitShift_15 = 15;
const uint16_t BitShift_16 = 16;
const uint16_t BitShift_20 = 20;
const uint16_t BitShift_24 = 24;
const uint16_t BitShift_27 = 27;
const uint16_t BitShift_28 = 28;
const uint16_t BitShift_31 = 31;
const uint16_t BitShift_32 = 32;
const uint16_t BitShift_36 = 36;
const uint16_t BitShift_40 = 40;
const uint16_t BitShift_44 = 44;
const uint16_t BitShift_48 = 48;
const uint16_t BitShift_52 = 52;
const uint16_t BitShift_56 = 56;
const uint16_t BitShift_59 = 59;
const uint16_t BitShift_60 = 60;
const uint16_t BitShift_63 = 63;
const uint16_t BitShift_64 = 64;
const uint16_t BitShift_128 = 128;
const uint16_t BitShift_255 = 255;
const uint16_t BitShift_256 = 256;
const uint16_t BitShift_512 = 512;
const uint16_t BitShift_768 = 768;
const uint16_t BitShift_784 = 784;
const uint16_t BitShift_1020 = 1020;
const uint16_t BitShift_1024 = 1024;
const uint16_t BitShift_3136 = 3136;
const uint16_t BitShift_4096 = 4096;
const uint16_t BitShift_6144 = 6144;
const uint16_t BitShift_10240 = 10240;

/**
 * @ingroup fp16 basic parameter
 * @brief   fp16 exponent bias
 */
#define FP16_EXP_BIAS (15)
/**
 * @ingroup fp16 basic parameter
 * @brief   the exponent bit length of fp16 is 5
 */
#define FP16_EXP_LEN (5)
/**
 * @ingroup fp16 basic parameter
 * @brief   the mantissa bit length of fp16 is 10
 */
#define FP16_MAN_LEN (10)
/**
 * @ingroup fp16 basic parameter
 * @brief   bit index of sign in fp16
 */
#define FP16_SIGN_INDEX (15)
/**
 * @ingroup fp16 basic parameter
 * @brief   sign mask of fp16         (1 00000 00000 00000)
 */
#define FP16_SIGN_MASK (0x8000)
/**
 * @ingroup fp16 basic parameter
 * @brief   exponent mask of fp16     (  11111 00000 00000)
 */
#define FP16_EXP_MASK (0x7C00)
/**
 * @ingroup fp16 basic parameter
 * @brief   mantissa mask of fp16     (        11111 11111)
 */
#define FP16_MAN_MASK (0x03FF)
/**
 * @ingroup fp16 basic parameter
 * @brief   conceal bit of mantissa of fp16(   1 00000 00000)
 */
#define FP16_MAN_HIDE_BIT (0x0400)
/**
 * @ingroup fp16 basic parameter
 * @brief   maximum value            (0111 1011 1111 1111)
 */
#define FP16_MAX (0x7BFF)
/**
 * @ingroup fp16 basic parameter
 * @brief   minimum value            (1111 1011 1111 1111)
 */
#define FP16_MIN (0xFBFF)
/**
 * @ingroup fp16 basic parameter
 * @brief   absolute maximum value   (0111 1111 1111 1111)
 */
#define FP16_ABS_MAX (0x7FFF)
/**
 * @ingroup fp16 basic parameter
 * @brief   maximum exponent value of fp16 is 15(11111)
 */
#define FP16_MAX_EXP (0x001F)
/**
 * @ingroup fp16 basic parameter
 * @brief   maximum valid exponent value of fp16 is 14(11110)
 */
#define FP16_MAX_VALID_EXP (0x001E)
/**
 * @ingroup fp16 basic parameter
 * @brief   maximum mantissa value of fp16(11111 11111)
 */
#define FP16_MAX_MAN (0x03FF)
/**
 * @ingroup fp16 basic parameter
 * @brief   absolute minimum normal value of fp16
 *         (E=1,M=0 D=2^(-14)=0.00006103515625)
 */
#define FP16_MIN_NORMAL ((1.0f / (2 << 14)))

/**
 * @ingroup fp16 basic operator
 * @brief   get sign of fp16
 */
#define FP16_EXTRAC_SIGN(x) (((x) >> 15) & 1)
/**
 * @ingroup fp16 basic operator
 * @brief   get exponent of fp16
 */
#define FP16_EXTRAC_EXP(x) (((x) >> 10) & FP16_MAX_EXP)
/**
 * @ingroup fp16 basic operator
 * @brief   get mantissa of fp16
 */
#define FP16_EXTRAC_MAN(x) ((((x) >> 0) & 0x3FF) | (((((x) >> 10) & 0x1F) > 0 ? 1 : 0) * 0x400))
/**
 * @ingroup fp16 basic operator
 * @brief   constructor of fp16 from sign exponent and mantissa
 */
#define FP16_CONSTRUCTOR(s, e, m) \
  (static_cast<uint16_t>(((s) << FP16_SIGN_INDEX) | ((e) << FP16_MAN_LEN) | ((m)&FP16_MAX_MAN)))

/**
 * @ingroup fp16 special value judgment
 * @brief   whether a fp16 is zero
 */
#define FP16_IS_ZERO(x) (((x)&FP16_ABS_MAX) == 0)
/**
 * @ingroup fp16 special value judgment
 * @brief   whether a fp16 is a denormalized value
 */
#define FP16_IS_DENORM(x) ((((x)&FP16_EXP_MASK) == 0))
/**
 * @ingroup fp16 special value judgment
 * @brief   whether a fp16 is infinite
 */
#define FP16_IS_INF(x) (((x)&FP16_ABS_MAX) == FP16_ABS_MAX)
/**
 * @ingroup fp16 special value judgment
 * @brief   whether a fp16 is NaN
 */
#define FP16_IS_NAN(x) (((x & FP16_EXP_MASK) == FP16_EXP_MASK) && (x & FP16_MAN_MASK))
/**
 * @ingroup fp16 special value judgment
 * @brief   whether a fp16 is invalid
 */
#define FP16_IS_INVALID(x) ((x & FP16_EXP_MASK) == FP16_EXP_MASK)
/**
 * @ingroup fp32 basic parameter
 * @brief   fp32 exponent bias
 */
#define FP32_EXP_BIAS (127)
/**
 * @ingroup fp32 basic parameter
 * @brief   the exponent bit length of float/fp32 is 8
 */
#define FP32_EXP_LEN (8)
/**
 * @ingroup fp32 basic parameter
 * @brief   the mantissa bit length of float/fp32 is 23
 */
#define FP32_MAN_LEN (23)
/**
 * @ingroup fp32 basic parameter
 * @brief   bit index of sign in float/fp32
 */
#define FP32_SIGN_INDEX (31)
/**
 * @ingroup fp32 basic parameter
 * @brief   sign mask of fp32         (1 0000 0000  0000 0000 0000 0000 000)
 */
#define FP32_SIGN_MASK (0x80000000u)
/**
 * @ingroup fp32 basic parameter
 * @brief   exponent mask of fp32     (  1111 1111  0000 0000 0000 0000 000)
 */
#define FP32_EXP_MASK (0x7F800000u)
/**
 * @ingroup fp32 basic parameter
 * @brief   mantissa mask of fp32     (             1111 1111 1111 1111 111)
 */
#define FP32_MAN_MASK (0x007FFFFFu)
/**
 * @ingroup fp32 basic parameter
 * @brief   conceal bit of mantissa of fp32      (  1  0000 0000 0000 0000 000)
 */
#define FP32_MAN_HIDE_BIT (0x00800000u)
/**
 * @ingroup fp32 basic parameter
 * @brief   absolute maximum value    (0 1111 1111  1111 1111 1111 1111 111)
 */
#define FP32_ABS_MAX (0x7FFFFFFFu)
/**
 * @ingroup fp32 basic parameter
 * @brief   maximum exponent value of fp32 is 255(1111 1111)
 */
#define FP32_MAX_EXP (0xFF)
/**
 * @ingroup fp32 basic parameter
 * @brief   maximum mantissa value of fp32    (1111 1111 1111 1111 1111 111)
 */
#define FP32_MAX_MAN (0x7FFFFF)
/**
 * @ingroup fp32 special value judgment
 * @brief   whether a fp32 is NaN
 */
#define FP32_IS_NAN(x) (((x & FP32_EXP_MASK) == FP32_EXP_MASK) && (x & FP32_MAN_MASK))
/**
 * @ingroup fp32 special value judgment
 * @brief   whether a fp32 is infinite
 */
#define FP32_IS_INF(x) (((x & FP32_EXP_MASK) == FP32_EXP_MASK) && (!(x & FP32_MAN_MASK)))
/**
 * @ingroup fp32 special value judgment
 * @brief   whether a fp32 is a denormalized value
 */
#define FP32_IS_DENORM(x) ((((x)&FP32_EXP_MASK) == 0))
/**
 * @ingroup fp32 basic operator
 * @brief   get sign of fp32
 */
#define FP32_EXTRAC_SIGN(x) (((x) >> FP32_SIGN_INDEX) & 1)
/**
 * @ingroup fp32 basic operator
 * @brief   get exponent of fp16
 */
#define FP32_EXTRAC_EXP(x) (((x)&FP32_EXP_MASK) >> FP32_MAN_LEN)
/**
 * @ingroup fp32 basic operator
 * @brief   get mantissa of fp16
 */
#define FP32_EXTRAC_MAN(x) \
  (((x)&FP32_MAN_MASK) | (((((x) >> FP32_MAN_LEN) & FP32_MAX_EXP) > 0 ? 1 : 0) * FP32_MAN_HIDE_BIT))
/**
 * @ingroup fp32 basic operator
 * @brief   constructor of fp32 from sign exponent and mantissa
 */
#define FP32_CONSTRUCTOR(s, e, m) (((s) << FP32_SIGN_INDEX) | ((e) << FP32_MAN_LEN) | ((m)&FP32_MAX_MAN))

/**
 * @ingroup fp64 basic parameter
 * @brief   fp64 exponent bias
 */
#define FP64_EXP_BIAS (1023)
/**
 * @ingroup fp64 basic parameter
 * @brief   the exponent bit length of double/fp64 is 11
 */
#define FP64_EXP_LEN (11)
/**
 * @ingroup fp64 basic parameter
 * @brief   the mantissa bit length of double/fp64 is 52
 */
#define FP64_MAN_LEN (52)
/**
 * @ingroup fp64 basic parameter
 * @brief   bit index of sign in double/fp64 is 63
 */
#define FP64_SIGN_INDEX (63)
/**
 * @ingroup fp64 basic parameter
 * @brief   sign mask of fp64                 (1 000                   (total 63bits 0))
 */
#define FP64_SIGN_MASK (0x8000000000000000LLu)
/**
 * @ingroup fp64 basic parameter
 * @brief   exponent mask of fp64            (0 1 11111 11111  0000?-?-(total 52bits 0))
 */
#define FP64_EXP_MASK (0x7FF0000000000000LLu)
/**
 * @ingroup fp64 basic parameter
 * @brief   mantissa mask of fp64            (                 1111?-?-(total 52bits 1))
 */
#define FP64_MAN_MASK (0x000FFFFFFFFFFFFFLLu)
/**
 * @ingroup fp64 basic parameter
 * @brief   conceal bit of mantissa of fp64     (               1 0000?-?-(total 52bits 0))
 */
#define FP64_MAN_HIDE_BIT (0x0010000000000000LLu)
/**
 * @ingroup fp64 basic parameter
 * @brief   absolute maximum value           (0 111?-?-(total 63bits 1))
 */
#define FP64_ABS_MAX (0x7FFFFFFFFFFFFFFFLLu)
/**
 * @ingroup fp64 basic parameter
 * @brief   maximum exponent value of fp64 is 2047(1 11111 11111)
 */
#define FP64_MAX_EXP (0x07FF)
/**
 * @ingroup fp64 basic parameter
 * @brief   maximum mantissa value of fp64  (111?-?-(total 52bits 1))
 */
#define FP64_MAX_MAN (0xFFFFFFFFFFFLLu)
/**
 * @ingroup fp64 special value judgment
 * @brief   whether a fp64 is NaN
 */
#define FP64_IS_NAN(x) (((x & FP64_EXP_MASK) == FP64_EXP_MASK) && (x & FP64_MAN_MASK))
/**
 * @ingroup fp64 special value judgment
 * @brief   whether a fp64 is infinite
 */
#define FP64_IS_INF(x) (((x & FP64_EXP_MASK) == FP64_EXP_MASK) && (!(x & FP64_MAN_MASK)))

/**
 * @ingroup integer special value judgment
 * @brief   maximum positive value of int8_t            (0111 1111)
 */
#define INT8_T_MAX (0x7F)
/**
 * @ingroup integer special value judgment
 * @brief   maximum value of a data with 8 bits length  (1111 111)
 */
#define BIT_LEN8_MAX (0xFF)
/**
 * @ingroup integer special value judgment
 * @brief   maximum positive value of int16_t           (0111 1111 1111 1111)
 */
#define INT16_T_MAX (0x7FFF)
/**
 * @ingroup integer special value judgment
 * @brief   maximum value of a data with 16 bits length (1111 1111 1111 1111)
 */
#define BIT_LEN16_MAX (0xFFFF)
/**
 * @ingroup integer special value judgment
 * @brief   maximum positive value of int32_t           (0111 1111 1111 1111 1111 1111 1111 1111)
 */
#define INT32_T_MAX (0x7FFFFFFFu)
/**
 * @ingroup integer special value judgment
 * @brief   maximum value of a data with 32 bits length (1111 1111 1111 1111 1111 1111 1111 1111)
 */
#define BIT_LEN32_MAX (0xFFFFFFFFu)
/**
 * @ingroup print switch
 * @brief   print an error if input fp16 is overflow
 */

/**
 * @ingroup fp16_t enum
 * @brief   round mode of last valid digital
 */
typedef enum tagFp16RoundMode {
  ROUND_TO_NEAREST = 0, /**< round to nearest even */
  ROUND_BY_TRUNCATED,   /**< round by truncated    */
  ROUND_MODE_RESERVED,
} fp16RoundMode_t;

/**
 * @ingroup fp16_t
 * @brief   Half precision float
 *         bit15:       1 bit SIGN      +---+-----+------------+
 *         bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
 *         bit0-9:      10bit MAN       +---+-----+------------+
 *
 */
typedef struct tagFp16 {
  uint16_t val;

 public:
  /**
   * @ingroup fp16_t constructor
   * @brief   Constructor without any param(default constructor)
   */
  tagFp16(void) {
    val = 0x0u;
  }
  /**
   * @ingroup all type constructor
   * @brief   Constructor with all type
   */
  template <typename T>
  tagFp16(const T& value) {
    *this = value;
  }
  /**
   * @ingroup fp16_t constructor
   * @brief   Constructor with an uint16_t value
   */
  tagFp16(const uint16_t& uiVal) : val(uiVal) {
  }
  /**
   * @ingroup fp16_t constructor
   * @brief   Constructor with a fp16_t object(copy constructor)
   */
  tagFp16(const tagFp16& fp) : val(fp.val) {
  }

  /**
   * @ingroup fp16_t math operator
   * @param [in] fp fp16_t object to be added
   * @brief   Override addition operator to performing fp16_t addition
   * @return  Return fp16_t result of adding this and fp
   */
  tagFp16 operator+(const tagFp16 fp);
  /**
   * @ingroup fp16_t math operator
   * @param [in] fp fp16_t object to be subtracted
   * @brief   Override addition operator to performing fp16_t subtraction
   * @return  Return fp16_t result of subtraction fp from this
   */
  tagFp16 operator-(const tagFp16 fp);
  /**
   * @ingroup fp16_t math operator
   * @param [in] fp fp16_t object to be multiplied
   * @brief   Override multiplication operator to performing fp16_t multiplication
   * @return  Return fp16_t result of multiplying this and fp
   */
  tagFp16 operator*(const tagFp16 fp);
  /**
   * @ingroup fp16_t math operator divided
   * @param [in] fp fp16_t object to be divided
   * @brief   Override division operator to performing fp16_t division
   * @return  Return fp16_t result of division this by fp
   */
  tagFp16 operator/(const tagFp16 fp);
  /**
   * @ingroup fp16_t math operator
   * @param [in] fp fp16_t object to be added
   * @brief   Override addition operator to performing fp16_t addition
   * @return  Return fp16_t result of adding this and fp
   */
  tagFp16 operator+=(const tagFp16 fp);
  /**
   * @ingroup fp16_t math operator
   * @param [in] fp fp16_t object to be subtracted
   * @brief   Override addition operator to performing fp16_t subtraction
   * @return  Return fp16_t result of subtraction fp from this
   */
  tagFp16 operator-=(const tagFp16 fp);
  /**
   * @ingroup fp16_t math operator
   * @param [in] fp fp16_t object to be multiplied
   * @brief   Override multiplication operator to performing fp16_t multiplication
   * @return  Return fp16_t result of multiplying this and fp
   */
  tagFp16 operator*=(const tagFp16 fp);
  /**
   * @ingroup fp16_t math operator divided
   * @param [in] fp fp16_t object to be divided
   * @brief   Override division operator to performing fp16_t division
   * @return  Return fp16_t result of division this by fp
   */
  tagFp16 operator/=(const tagFp16 fp);

  /**
   * @ingroup fp16_t math compare operator
   * @param [in] fp fp16_t object to be compared
   * @brief   Override basic comparison operator to performing fp16_t if-equal comparison
   * @return  Return boolean result of if-equal comparison of this and fp.
   */
  bool operator==(const tagFp16& fp) const;
  /**
   * @ingroup fp16_t math compare operator
   * @param [in] fp fp16_t object to be compared
   * @brief   Override basic comparison operator to performing fp16_t not-equal comparison
   * @return  Return boolean result of not-equal comparison of this and fp.
   */
  bool operator!=(const tagFp16& fp) const;
  /**
   * @ingroup fp16_t math compare operator
   * @param [in] fp fp16_t object to be compared
   * @brief   Override basic comparison operator to performing fp16_t greater-than comparison
   * @return  Return boolean result of greater-than comparison of this and fp.
   */
  bool operator>(const tagFp16& fp) const;
  /**
   * @ingroup fp16_t math compare operator
   * @param [in] fp fp16_t object to be compared
   * @brief   Override basic comparison operator to performing fp16_t greater-equal comparison
   * @return  Return boolean result of greater-equal comparison of this and fp.
   */
  bool operator>=(const tagFp16& fp) const;
  /**
   * @ingroup fp16_t math compare operator
   * @param [in] fp fp16_t object to be compared
   * @brief   Override basic comparison operator to performing fp16_t less-than comparison
   * @return  Return boolean result of less-than comparison of this and fp.
   */
  bool operator<(const tagFp16& fp) const;
  /**
   * @ingroup fp16_t math compare operator
   * @param [in] fp fp16_t object to be compared
   * @brief   Override basic comparison operator to performing fp16_t less-equal comparison
   * @return  Return boolean result of less-equal comparison of this and fp.
   */
  bool operator<=(const tagFp16& fp) const;

  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] fp fp16_t object to be copy to fp16_t
   * @brief   Override basic evaluation operator to copy fp16_t to a new fp16_t
   * @return  Return fp16_t result from fp
   */
  tagFp16& operator=(const tagFp16& fp);
  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] fVal float object to be converted to fp16_t
   * @brief   Override basic evaluation operator to convert float to fp16_t
   * @return  Return fp16_t result from fVal
   */
  tagFp16& operator=(const float& fVal);
  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] dVal double object to be converted to fp16_t
   * @brief   Override basic evaluation operator to convert double to fp16_t
   * @return  Return fp16_t result from dVal
   */
  tagFp16& operator=(const double& dVal);
  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] iVal float object to be converted to fp16_t
   * @brief   Override basic evaluation operator to convert float to fp16_t
   * @return  Return fp16_t result from iVal
   */
  tagFp16& operator=(const int8_t& iVal);
  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] uiVal uint8_t object to be converted to fp16_t
   * @brief   Override basic evaluation operator to convert uint8_t to fp16_t
   * @return  Return fp16_t result from uiVal
   */
  tagFp16& operator=(const uint8_t& uiVal);
  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] iVal int16_t object to be converted to fp16_t
   * @brief   Override basic evaluation operator to convert int16_t to fp16_t
   * @return  Return fp16_t result from iVal
   */
  tagFp16& operator=(const int16_t& iVal);
  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] uiVal uint16_t object to be converted to fp16_t
   * @brief   Override basic evaluation operator to convert uint16_t to fp16_t
   * @return  Return fp16_t result from uiVal
   */
  tagFp16& operator=(const uint16_t& uiVal);
  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] iVal int32_t object to be converted to fp16_t
   * @brief   Override basic evaluation operator to convert int32_t to fp16_t
   * @return  Return fp16_t result from iVal
   */
  tagFp16& operator=(const int32_t& iVal);
  /**
   * @ingroup fp16_t math evaluation operator
   * @param [in] uiVal uint32_t object to be converted to fp16_t
   * @brief   Override basic evaluation operator to convert uint32_t to fp16_t
   * @return  Return fp16_t result from uiVal
   */
  tagFp16& operator=(const uint32_t& uiVal);
  /**
   * @ingroup fp16_t math conversion
   * @brief   Override convert operator to convert fp16_t to float/fp32
   * @return  Return float/fp32 value of fp16_t
   */
  operator float() const;
  /**
   * @ingroup fp16_t math conversion
   * @brief   Override convert operator to convert fp16_t to double/fp64
   * @return  Return double/fp64 value of fp16_t
   */
  operator double() const;
  /**
   * @ingroup fp16_t math conversion
   * @brief   Override convert operator to convert fp16_t to int8_t
   * @return  Return int8_t value of fp16_t
   */
  operator int8_t() const;
  /**
   * @ingroup fp16_t math conversion
   * @brief   Override convert operator to convert fp16_t to uint8_t
   * @return  Return uint8_t value of fp16_t
   */
  operator uint8_t() const;
  /**
   * @ingroup fp16_t conversion
   * @brief   Override convert operator to convert fp16_t to int16_t
   * @return  Return int16_t value of fp16_t
   */
  operator int16_t() const;
  /**
   * @ingroup fp16_t math conversion
   * @brief   Override convert operator to convert fp16_t to uint16_t
   * @return  Return uint16_t value of fp16_t
   */
  operator uint16_t() const;
  /**
   * @ingroup fp16_t math conversion
   * @brief   Override convert operator to convert fp16_t to int32_t
   * @return  Return int32_t value of fp16_t
   */
  operator int32_t() const;
  /**
   * @ingroup fp16_t math conversion
   * @brief   Override convert operator to convert fp16_t to int64_t
   * @return  Return int64_t value of fp16_t
   */
  operator uint32_t() const;
  /**
   * @ingroup fp16_t judgment method
   * @param [in] fp fp16_t object to be judgement
   * @brief   whether a fp16_t is inifinite
   * @return  Returns 1:+INF -1:-INF 0:not INF
   */
  int IsInf();

  /**
   * @ingroup fp16_t math conversion
   * @brief   Convert fp16_t to float/fp32
   * @return  Return float/fp32 value of fp16_t
   */
  float toFloat();
  /**
   * @ingroup fp16_t math conversion
   * @brief   Convert fp16_t to double/fp64
   * @return  Return double/fp64 value of fp16_t
   */
  double toDouble();
  /**
   * @ingroup fp16_t math conversion
   * @brief   Convert fp16_t to int8_t
   * @return  Return int8_t value of fp16_t
   */
  int8_t toInt8();
  /**
   * @ingroup fp16_t math conversion
   * @brief   Convert fp16_t to uint8_t
   * @return  Return uint8_t value of fp16_t
   */
  uint8_t toUInt8();
  /**
   * @ingroup fp16_t conversion
   * @brief   Convert fp16_t to int16_t
   * @return  Return int16_t value of fp16_t
   */
  int16_t toInt16();
  /**
   * @ingroup fp16_t math conversion
   * @brief   Convert fp16_t to uint16_t
   * @return  Return uint16_t value of fp16_t
   */
  uint16_t toUInt16();
  /**
   * @ingroup fp16_t math conversion
   * @brief   Convert fp16_t to int32_t
   * @return  Return int32_t value of fp16_t
   */
  int32_t toInt32();
  /**
   * @ingroup fp16_t math conversion
   * @brief   Convert fp16_t to int64_t
   * @return  Return int64_t value of fp16_t
   */
  uint32_t toUInt32();

} fp16_t;

/**
 * @ingroup fp16_t public method
 * @param [in]     val signature is negative
 * @param [in|out] s   sign of fp16_t object
 * @param [in|out] e   exponent of fp16_t object
 * @param [in|out] m   mantissa of fp16_t object
 * @brief   Extract the sign, exponent and mantissa of a fp16_t object
 */
void ExtractFP16(const uint16_t& val, uint16_t* s, int16_t* e, uint16_t* m);
/**
 * @ingroup fp16_t public method
 * @param [in]     negative sign is negative
 * @param [in|out] man      mantissa to be reverse
 * @brief   Calculate a mantissa's complement (add ont to it's radix-minus-one complement)
 * @return  Return complement of man
 */
template <typename T>
void ReverseMan(bool negative, T* man) {
  if (negative) {
    *man = (~(*man)) + 1;
  }
}
/**
 * @ingroup fp16_t public method
 * @param [in] ea exponent of one fp16_t/float number
 * @param [in] ma mantissa of one fp16_t/float number
 * @param [in] eb exponent of another fp16_t/float number
 * @param [in] mb mantissa of another fp16_t/float number
 * @brief   choose mantissa to be shift right whoes exponent is less than another one
 * @return  Return mantissawhoes exponent is less than another one
 */
template <typename T>
T* MinMan(const int16_t& ea, T* ma, const int16_t& eb, T* mb) {
  return (ea > eb) ? mb : ma;
}
/**
 * @ingroup fp16_t public method
 * @param [in] man   mantissa to be operate
 * @param [in] shift right shift bits
 * @brief   right shift a mantissa
 * @return  Return right-shift mantissa
 */
/*lint -e1573*/
template <typename T>
T RightShift(T man, int16_t shift) {
  int bits = sizeof(T) * 8;
  T mask = (((T)1u) << ((unsigned int)(bits - 1)));
  for (int i = 0; i < shift; i++) {
    man = ((man & mask) | (man >> 1));
  }
  return man;
}
/*lint +e1573*/
/**
 * @ingroup fp16_t public method
 * @param [in] ea exponent of one temp fp16_t number
 * @param [in] ma mantissa of one temp fp16_t number
 * @param [in] eb exponent of another temp fp16_t number
 * @param [in] mb mantissa of another temp fp16_t number
 * @brief   Get mantissa sum of two temp fp16_t numbers, T support types: uint16_t/uint32_t/uint64_t
 * @return  Return mantissa sum
 */
/*lint -e1573*/
template <typename T>
T GetManSum(int16_t ea, const T& ma, int16_t eb, const T& mb) {
  T sum = 0;
  if (ea != eb) {
    T m_tmp = 0;
    int16_t e_tmp = std::abs(ea - eb);
    if (ea > eb) {
      m_tmp = mb;
      m_tmp = RightShift(m_tmp, e_tmp);
      sum = ma + m_tmp;
    } else {
      m_tmp = ma;
      m_tmp = RightShift(m_tmp, e_tmp);
      sum = m_tmp + mb;
    }
  } else {
    sum = ma + mb;
  }
  return sum;
}
/*lint +e1573*/
/**
 * @ingroup fp16_t public method
 * @param [in] bit0    whether the last preserved bit is 1 before round
 * @param [in] bit1    whether the abbreviation's highest bit is 1
 * @param [in] bitLeft whether the abbreviation's bits which not contain highest bit grater than 0
 * @param [in] man     mantissa of a fp16_t or float number, support types: uint16_t/uint32_t/uint64_t
 * @param [in] shift   abbreviation bits
 * @brief    Round fp16_t or float mantissa to nearest value
 * @return   Returns true if round 1,otherwise false;
 */
/*lint -e1573*/
template <typename T>
T ManRoundToNearest(bool bit0, bool bit1, bool bitLeft, T man, uint16_t shift = 0) {
  man = (man >> shift) + ((bit1 && (bitLeft || bit0)) ? 1 : 0);
  return man;
}
/*lint +e1573*/
/**
 * @ingroup fp16_t public method
 * @param [in] man    mantissa of a float number, support types: uint16_t/uint32_t/uint64_t
 * @brief   Get bit length of a uint32_t number
 * @return  Return bit length of man
 */
/*lint -e1573*/
template <typename T>
int16_t GetManBitLength(T man) {
  int16_t len = 0;
  while (man) {
    man >>= 1;
    len++;
  }
  return len;
}
/*lint +e1573*/
};  // namespace fe

#endif /*_FP16_T_HPP_*/
