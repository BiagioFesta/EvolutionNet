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
#ifndef EVOLUTION_NET_TYPES_HPP
#define EVOLUTION_NET_TYPES_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

namespace EvolutionNet {

using NodeId = std::uint32_t;
using ConnectionHash = std::uint64_t;
using LayerId = unsigned int;
using RndEngine = std::mt19937_64;
using FitnessScore = float;

struct DefaultParamConfig {
  static constexpr float ProbMutationWeight = 0.8f;
  static constexpr float ProbMutationWeightPerturbation = 0.9f;
  static constexpr float ProbMutatationNewNode = 0.02f;
  static constexpr float ProbMutationNewConnection = 0.05f;
  static constexpr float ProbInheritOnFitterGenome = 0.5f;
  static constexpr float ProbInheritDisabledGene = 0.75f;
  static constexpr float SpeciesSimilarityThreshold = 4.0f;
  static constexpr int NormalizedSizeGene = 20;
  static constexpr float SimilarityCoefExcess = 1.f;
  static constexpr float SimilarityCoefDisj = 1.f;
  static constexpr float SimilarityCoefWeight = 3.f;
  static constexpr float ProbOffspringCrossover = 0.75f;
  static constexpr float ProbMatingInterspecies = 0.001f;
  static constexpr std::size_t SizeSpecieForChampion = 5;
  static constexpr std::size_t GenForStagningSpecies = 15;
  static inline std::uniform_real_distribution<float> DistributionNewWeight{
      -1.f,
      1.f};
  static inline std::normal_distribution<float> DistributionPertWeight{0.f,
                                                                       1.f};

  static float WeightPerturbation(float weight, RndEngine* rndEngine) noexcept {
    weight += DistributionPertWeight(*rndEngine) / 50;
    return std::clamp(weight, -1.f, 1.f);
  }

  static float NewRndWeight(RndEngine* rndEngine) noexcept {
    return DistributionNewWeight(*rndEngine);
  }

  static float ActivationFn(float value) noexcept {
    return 1.f / (1.f + std::exp(-4.9f * value));
  }
};

}  // namespace EvolutionNet

#endif  // EVOLUTION_NET_TYPES_HPP
