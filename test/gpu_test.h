#pragma once

#include "test_support.h"

#include <gtest/gtest.h>

namespace gs_native::test {

class GpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!HasCuda()) {
            GTEST_SKIP() << "A CUDA device is required.";
        }
    }
}; // class GpuTest

} // namespace gs_native::test
