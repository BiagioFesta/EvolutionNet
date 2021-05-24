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
#ifndef EVOLUTION_NET_TESTS_RND_NUMBER_GENERATOR_HPP
#define EVOLUTION_NET_TESTS_RND_NUMBER_GENERATOR_HPP
#include <gtest/gtest.h>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace EvolutionNet::Tests {

class RndEngine final {
 public:
  using InternalEngine = std::mt19937_64;
  using Seed = InternalEngine::result_type;

  explicit RndEngine(const Seed seed) : engine_(seed) {}

  inline InternalEngine* getInternalEngine() noexcept {
    return &engine_;
  }

  static RndEngine GenerateRndEngine() {
    using Seed = RndEngine::Seed;
    static constexpr const char* EVOLUTION_NET_SEED = "EVOLUTION_NET_TEST_SEED";

    const Seed rndSeed = static_cast<Seed>(std::random_device()());
    const char* const env = std::getenv(EVOLUTION_NET_SEED);
    const Seed seed = env == nullptr ? rndSeed : static_cast<Seed>(std::stoull(env));

    std::cout << "[          ]"
              << "   RndSeed: " << seed << '\n';
    return RndEngine(seed);
  }

 private:
  InternalEngine engine_;
};

template <typename Integral>
class RndIntegral final {
 public:
  static_assert(std::is_integral_v<Integral>);
  using Value = Integral;

  inline Value generate(RndEngine* rndEngine) {
    return rndInt_(*rndEngine->getInternalEngine());
  }

 private:
  std::uniform_int_distribution<Integral> rndInt_;
};

template <typename Integral>
class RndUniqueSeqIntegral final {
 public:
  static_assert(std::is_integral_v<Integral>);
  using Value = Integral;

  RndUniqueSeqIntegral(const std::size_t len, RndEngine* rndEngine) : seq_(len) {
    std::iota(seq_.begin(), seq_.end(), Integral{});
    std::shuffle(seq_.begin(), seq_.end(), *rndEngine->getInternalEngine());
  }

  inline Value pop() noexcept {
    assert(!seq_.empty());
    const Value value = seq_.back();
    seq_.pop_back();
    return value;
  }

 private:
  std::vector<Integral> seq_;
};

}  // namespace EvolutionNet::Tests

#endif  // EVOLUTION_NET_TESTS_RND_NUMBER_GENERATOR_HPP
