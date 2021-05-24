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
#include <EvolutionNet/FlatMap.hpp>
#include <string>
#include <utility>
#include "RndNumberGenerator.hpp"

namespace EvolutionNet::Tests {

class FlatMapTest : public ::testing::Test {
 public:
  using Key = int;
  using Value = std::string;
  using FlatMap = EvolutionNet::FlatMap<Key, Value>;
  using RndKey = RndIntegral<Key>;
  using RndUniqueSeqKey = RndUniqueSeqIntegral<Key>;
  static constexpr int SequenceLen = 42;

  static Value Key2Value(const Key key) {
    return std::to_string(key);
  }

  FlatMap flatMap_;
};

TEST_F(FlatMapTest, DefaultEmpty) {
  ASSERT_EQ(flatMap_.size(), 0ull);
  ASSERT_TRUE(flatMap_.valuesVector().empty());
  ASSERT_TRUE(std::as_const(flatMap_).valuesVector().empty());
}

TEST_F(FlatMapTest, InsertOneValue) {
  auto rndEngine = RndEngine::GenerateRndEngine();
  const Key key = RndKey().generate(&rndEngine);
  const Value value = Key2Value(key);

  Value* const insertedValue = flatMap_.insert(key, value);

  ASSERT_NE(insertedValue, nullptr);
  ASSERT_EQ(*insertedValue, value);
  ASSERT_EQ(flatMap_.size(), 1ull);
  ASSERT_EQ(flatMap_.valuesVector().size(), 1ull);
  ASSERT_EQ(std::as_const(flatMap_).valuesVector().size(), 1ull);
  ASSERT_EQ(flatMap_.valuesVector().front(), value);
  ASSERT_EQ(&(flatMap_.valuesVector().front()), insertedValue);
  ASSERT_EQ(flatMap_[key], insertedValue);
  ASSERT_EQ(std::as_const(flatMap_)[key], insertedValue);
  ASSERT_EQ(flatMap_.nthKey(0), key);
  ASSERT_EQ(flatMap_.nthValue(0), value);
  ASSERT_EQ(std::as_const(flatMap_).nthValue(0), value);
  ASSERT_EQ(&flatMap_.nthValue(0), insertedValue);
  ASSERT_EQ(&std::as_const(flatMap_).nthValue(0), insertedValue);
}

TEST_F(FlatMapTest, Clear) {
  auto rndEngine = RndEngine::GenerateRndEngine();
  const Key key = RndKey().generate(&rndEngine);
  const Value value = Key2Value(key);

  flatMap_.insert(key, value);
  flatMap_.clear();

  ASSERT_EQ(flatMap_.size(), 0ull);
  ASSERT_TRUE(flatMap_.valuesVector().empty());
  ASSERT_TRUE(std::as_const(flatMap_).valuesVector().empty());
}

TEST_F(FlatMapTest, InsertSequence) {
  auto rndEngine = RndEngine::GenerateRndEngine();
  RndUniqueSeqKey rndSeqKey(SequenceLen, &rndEngine);

  for (std::size_t i = 0; i < SequenceLen; ++i) {
    const Key key = rndSeqKey.pop();
    const Value value = Key2Value(key);

    ASSERT_EQ(flatMap_[key], nullptr);
    ASSERT_EQ(std::as_const(flatMap_)[key], nullptr);

    Value* const insertedValue = flatMap_.insert(key, value);

    ASSERT_EQ(flatMap_.size(), (i + 1));
    ASSERT_EQ(flatMap_[key], insertedValue);
    ASSERT_EQ(std::as_const(flatMap_)[key], insertedValue);

    std::vector<Key> insertedKeys(i + 1);
    for (std::size_t j = 0; j <= i; ++j) {
      insertedKeys[j] = flatMap_.nthKey(j);
      ASSERT_EQ(flatMap_.nthValue(j), Key2Value(insertedKeys[j]));
      ASSERT_EQ(std::as_const(flatMap_).nthValue(j), Key2Value(insertedKeys[j]));
    }
    ASSERT_TRUE(std::is_sorted(insertedKeys.cbegin(), insertedKeys.cend()));

    ASSERT_EQ(flatMap_.valuesVector().size(), (i + 1));
    ASSERT_EQ(std::as_const(flatMap_).valuesVector().size(), (i + 1));

    ASSERT_EQ(flatMap_.valuesVector().back(), value);
    ASSERT_EQ(std::as_const(flatMap_).valuesVector().back(), value);
  }
}

}  // namespace EvolutionNet::Tests
