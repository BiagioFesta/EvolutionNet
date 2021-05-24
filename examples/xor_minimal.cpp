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
#include <EvolutionNet/EvolutionNet.hpp>
#include <cmath>
#include <iostream>

static constexpr int NumInput = 2;
static constexpr int NumOutput = 1;
static constexpr bool Bias = true;
static constexpr float ThresholdFitness = 0.80f;
static constexpr std::size_t PopulationSize = 100;
using ParamConfig = EvolutionNet::DefaultParamConfig;
using EvolutionNetT = EvolutionNet::EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>;
using Network = EvolutionNetT::NetworkT;
using FitnessScore = EvolutionNet::FitnessScore;

static const std::vector<std::tuple<float, float, float>> dataset = {{0.f, 0.f, 0.f},
                                                                     {0.f, 1.f, 1.f},
                                                                     {1.f, 0.f, 1.f},
                                                                     {1.f, 1.f, 0.f}};

int main() {
  EvolutionNetT evolutionNet;

  evolutionNet.initialize(PopulationSize);

  while (true) {
    evolutionNet.evaluateAll([](Network* network) {
      FitnessScore score = 0.f;

      for (const auto& [x1, x2, y] : dataset) {
        network->setInputValue(0, x1);
        network->setInputValue(1, x2);

        network->feedForward<ParamConfig>();

        const float output = network->getOutputValue(0);
        assert(output >= 0.f && output <= 1.f);

        score += 1.f - (std::abs(output - y));
      }

      score /= static_cast<float>(dataset.size());

      network->setFitness(score);
    });

    if (evolutionNet.getBestFitness() >= ::ThresholdFitness) {
      break;
    }

    evolutionNet.evolve();
  }

  auto finalNetwork = evolutionNet.getBestNetworkMutable();
  for (const auto& [x1, x2, _] : ::dataset) {
    finalNetwork->setInputValue(0, x1);
    finalNetwork->setInputValue(1, x2);
    finalNetwork->feedForward<ParamConfig>();

    std::cout << "x1: " << x1 << ", x2: " << x2 << "  -->  " << finalNetwork->getOutputValue(0) << "\n";
  }
}
