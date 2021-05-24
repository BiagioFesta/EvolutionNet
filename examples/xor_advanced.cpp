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
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {

struct XorParamConfig : EvolutionNet::DefaultParamConfig {
  static constexpr float ProbMutationWeight = 0.03f;
  static constexpr float ProbMutatationNewNode = 0.01f;
  static constexpr float ProbMutationNewConnection = 0.1f;
};

constexpr int NumInput = 2;
constexpr int NumOutput = 1;
constexpr bool Bias = true;
constexpr float ThresholdFitness = 0.9999f;
constexpr std::size_t PopulationSize = 10000;
constexpr const char* NetworkFilename = "xor.enet";
using EvolutionNetT = EvolutionNet::EvolutionNet<NumInput, NumOutput, Bias, XorParamConfig>;
using Network = EvolutionNetT::NetworkT;
using FitnessScore = EvolutionNet::FitnessScore;
using Seed = EvolutionNetT::SeedT;

const std::vector<std::tuple<float, float, float>> dataset = {{0.f, 0.f, 0.f},
                                                              {0.f, 1.f, 1.f},
                                                              {1.f, 0.f, 1.f},
                                                              {1.f, 1.f, 0.f}};

void EvaluateNetwork(Network* network) {
  FitnessScore score = 0.f;

  for (const auto& [x1, x2, y] : dataset) {
    network->setInputValue(0, x1);
    network->setInputValue(1, x2);

    network->feedForward<XorParamConfig>();

    const float output = network->getOutputValue(0);
    assert(output >= 0.f && output <= 1.f);

    score += 1.f - (std::abs(output - y));
  }

  score /= static_cast<float>(dataset.size());

  network->setFitness(score);
}

void SaveNetworkOnFile(const Network& network) {
  std::cout << "Trying saving network on file '" << NetworkFilename << "'\n";
  std::ofstream file(NetworkFilename, std::ios_base::out | std::ios_base::binary);
  if (file.fail()) {
    std::cerr << "  Cannot save the network on file '" << NetworkFilename << "'\n";
    return;
  }

  network.serialize(&file);
  file.close();
  std::cout << "  Saved!\n";
}

bool TryLoadNetworkFromFile(Network* network) {
  std::cout << "Trying loading saved network from file '" << NetworkFilename << "'\n";
  std::ifstream file(NetworkFilename, std::ios_base::in | std::ios_base::binary);
  if (file.fail()) {
    std::cout << "  Cannot read the network on file '" << NetworkFilename << "'\n";
    std::cout << "  Computing...\n";
    return false;
  }

  if (!network->deserialize(&file)) {
    std::cout << "  Network on file '" << NetworkFilename << "' seems to be corrupted\n";
    std::cout << "  Computing...\n";
    return false;
  }

  std::cout << "  Network loaded!\n";
  return true;
}

Seed GenerateSeed() {
#ifdef NDEBUG
  return std::chrono::system_clock::now().time_since_epoch().count();
#else
  return 0;
#endif
}

}  // anonymous namespace

int main() {
  ::EvolutionNetT evolutionNet;
  ::Network savedNetwork;
  ::Network* finalNetwork;
  ::Seed rndSeed;

  if (::TryLoadNetworkFromFile(&savedNetwork)) {
    finalNetwork = &savedNetwork;
  } else {
    rndSeed = ::GenerateSeed();
    std::cout << "Random Seed: " << rndSeed << '\n';
    std::cout << "Target fitness: " << ::ThresholdFitness << '\n';
    evolutionNet.initialize(::PopulationSize, rndSeed);

    while (true) {
      evolutionNet.evaluateAll(EvaluateNetwork);

      std::cout << "\r                                                         \r";
      std::cout << "Fitness Current Generation (" << evolutionNet.getCounterGeneration()
                << "): " << evolutionNet.getBestFitness();
      std::cout.flush();

      if (evolutionNet.getBestFitness() >= ::ThresholdFitness) {
        std::cout << '\n';
        break;
      }

      evolutionNet.evolve();
    }

    finalNetwork = evolutionNet.getBestNetworkMutable();
    SaveNetworkOnFile(*finalNetwork);
  }

  for (const auto& [x1, x2, _] : ::dataset) {
    finalNetwork->setInputValue(0, x1);
    finalNetwork->setInputValue(1, x2);
    finalNetwork->feedForward<XorParamConfig>();

    std::cout << "x1: " << x1 << ", x2: " << x2 << "  -->  " << finalNetwork->getOutputValue(0) << "\n";
  }
}
