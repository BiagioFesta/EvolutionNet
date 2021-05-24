/*

  Copyright (C) 2021  Biagio Festa

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/
#include <gtest/gtest.h>
#include <EvolutionNet/FlatSet.hpp>
#include <iostream>
#include <unordered_set>
#include "RndNumberGenerator.hpp"

namespace EvolutionNet::Tests {

class FlatSetTest : public ::testing::Test {
 public:
  using Value = int;
  using FlatSet = EvolutionNet::FlatSet<Value>;
  using RndValue = RndIntegral<Value>;
  using RndUniqueSeqValue = RndUniqueSeqIntegral<int>;
  static constexpr int SequenceLen = 42;

  FlatSet flatSet_;
};

TEST_F(FlatSetTest, DefaultEmpty) {
  ASSERT_TRUE(flatSet_.empty());
  ASSERT_EQ(flatSet_.size(), 0ull);
}

TEST_F(FlatSetTest, OneInsert) {
  auto rndEngine = RndEngine::GenerateRndEngine();
  const Value value = RndValue().generate(&rndEngine);

  flatSet_.insert(value);

  ASSERT_FALSE(flatSet_.empty());
  ASSERT_EQ(flatSet_.size(), 1ull);
  ASSERT_EQ(*flatSet_.begin(), value);
  ASSERT_EQ(*(--flatSet_.end()), value);
  ASSERT_EQ(flatSet_.nth(0), value);
}

TEST_F(FlatSetTest, Clear) {
  auto rndEngine = RndEngine::GenerateRndEngine();
  const Value value = RndValue().generate(&rndEngine);

  flatSet_.insert(value);
  flatSet_.clear();

  ASSERT_TRUE(flatSet_.empty());
}

#ifndef NDEBUG
TEST_F(FlatSetTest, SameInsert) {
  auto rndEngine = RndEngine::GenerateRndEngine();
  const Value value = RndValue().generate(&rndEngine);

  flatSet_.insert(value);

  ASSERT_DEBUG_DEATH(flatSet_.insert(value), "\\*bound != value");
}
#endif

TEST_F(FlatSetTest, InsertSequence) {
  auto rndEngine = RndEngine::GenerateRndEngine();
  RndUniqueSeqValue rndSeq(SequenceLen, &rndEngine);

  for (int i = 0; i < SequenceLen; ++i) {
    const Value value = rndSeq.pop();

    flatSet_.insert(value);

    ASSERT_EQ(flatSet_.size(), static_cast<std::size_t>(i + 1));
    ASSERT_NE(std::find(flatSet_.begin(), flatSet_.end(), value), flatSet_.end());
    ASSERT_EQ(std::count(flatSet_.begin(), flatSet_.end(), value), 1u);
    ASSERT_TRUE(std::is_sorted(flatSet_.begin(), flatSet_.end()));
  }
}

}  // namespace EvolutionNet::Tests
