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
#ifndef EVOLUTION_NET_EVOLUTION_NET_HPP
#define EVOLUTION_NET_EVOLUTION_NET_HPP

#include <EvolutionNet/Genome.hpp>
#include <EvolutionNet/Network.hpp>
#include <EvolutionNet/Population.hpp>
#include <algorithm>
#include <cassert>
#include <type_traits>

namespace EvolutionNet {

/*! \class EvolutionNet
 *  \brief This class represents the main interface of this library.
 *  Some of the network features are set at compile time with template logic.
 *  The usage is the following:
 *     EvolutionNet<NumInput, NumOutput, Bias, Configuration> net;   // Constructor. Construct the object.
 *     net.initialize(populationSize, rndSeed);                      // Right after! Initialize the object.
 *
 *     while (!stopCriteria) {
 *       net.evaluateAll(EvaluateFn);                                // For each network in the population set fitness.
 *       net.evolve();                                               // Evolve in the next generation and loop!
 *     }
 */
template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
class EvolutionNet {
 public:
  using GenomeT = Genome<NumInput, NumOutput, Bias>;
  using PopulationT = Population<GenomeT>;
  using NetworkT = Network<GenomeT>;
  using SeedT = std::remove_cv_t<decltype(RndEngine::default_seed)>;
  static constexpr SeedT DefaultSeed = RndEngine::default_seed;

  /*! \brief Initialize the Evolution Net.
   *  You need to call this function before any other operation.
   *  Here you can set the size of the population and the random seed used internally.
   *  \note `populationSize` must be greater than zero, otherwise Undefined Behavior.
   */
  void initialize(const std::size_t populationSize, const SeedT rndSeed = DefaultSeed);

  //! \return all the networks (size of the population).
  inline std::vector<NetworkT>& getNetworks() noexcept;

  //! \return the n-th network of the population.
  inline NetworkT& getNetworkNth(const std::size_t i) noexcept;

  /*! \brief Evaluate all networks in the population.
   *  The `fn` function should assign a fitness score to all population.
   */
  template <typename Fn>
  inline void evaluateAll(Fn&& fn);

  //! \return the best fitness score for this current generation. Call this only after ending evaluation.
  inline FitnessScore getBestFitness() const noexcept;

  //! \return the best network accordinlying with the max fitness score. Call this only after ending evaluation.
  inline const NetworkT& getBestNetwork() const noexcept;
  inline NetworkT* getBestNetworkMutable() noexcept;

  /*! \brief Evolve the population.
   *  Call this only after ending evaluation, thus, fitness for each network has been set.
   */
  inline void evolve();

  //! \return the size of the population. This is constant (does not change during evolution).
  inline std::size_t getPopulationSize() const noexcept;

  //! \return the generation counter. First generation stats from 0.
  inline std::size_t getCounterGeneration() const noexcept;

 private:
  template <typename Fn, typename = std::void_t<>>
  struct IsValidEvaluationFunction : std::false_type {};

  template <typename Fn>
  struct IsValidEvaluationFunction<Fn, std::void_t<decltype(std::declval<Fn>()(std::declval<NetworkT*>()))>>
      : std::true_type {};

  RndEngine rndEngine_;
  PopulationT population_;
  std::vector<NetworkT> networks_;
  NetworkT* bestNetwork_ = nullptr;
  std::size_t counterGeneration_ = 0;

  inline void computeNetworks();
};

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
void EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::initialize(const std::size_t populationSize,
                                                                      const SeedT rndSeed) {
  assert(populationSize > 0);

  rndEngine_.seed(rndSeed);

  population_.template initialize<ParamConfig>(populationSize, &rndEngine_);

  networks_.resize(populationSize);
  computeNetworks();

  bestNetwork_ = nullptr;
  counterGeneration_ = 0;
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
inline std::vector<typename EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::NetworkT>&
EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::getNetworks() noexcept {
  return networks_;
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
inline typename EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::NetworkT&
EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::getNetworkNth(const std::size_t i) noexcept {
  assert(i < networks_.size());
  return networks_[i];
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
template <typename Fn>
inline void EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::evaluateAll(Fn&& fn) {
  static_assert(IsValidEvaluationFunction<Fn>::value,
                "Fn is not a valid evaluation function. It should be something like `void(*)(Network*)`");

  const std::size_t numNetworks = networks_.size();
  FitnessScore fitness, fitnessMax;

  assert(numNetworks > 0);
  assert(numNetworks == population_.getPopulationSize());

  fn(&(networks_[0]));
  bestNetwork_ = &(networks_[0]);
  fitnessMax = bestNetwork_->getFitness();
  population_.assignFitness(0, fitnessMax);

  for (std::size_t i = 1; i < numNetworks; ++i) {
    fn(&(networks_[i]));

    fitness = networks_[i].getFitness();
    population_.assignFitness(i, fitness);

    if (fitnessMax < fitness) {
      bestNetwork_ = &(networks_[i]);
      fitnessMax = fitness;
    }
  }
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
FitnessScore EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::getBestFitness() const noexcept {
  assert(bestNetwork_ != nullptr);
  return bestNetwork_->getFitness();
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
const typename EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::NetworkT&
EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::getBestNetwork() const noexcept {
  assert(bestNetwork_ != nullptr);
  return *bestNetwork_;
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
typename EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::NetworkT*
EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::getBestNetworkMutable() noexcept {
  assert(bestNetwork_ != nullptr);
  return bestNetwork_;
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
inline void EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::evolve() {
  assert(bestNetwork_ != nullptr);
  population_.template nextGeneration<ParamConfig>(&rndEngine_);
  computeNetworks();
  bestNetwork_ = nullptr;
  ++counterGeneration_;
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
inline std::size_t EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::getPopulationSize() const noexcept {
  return population_.getPopulationSize();
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
inline std::size_t EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::getCounterGeneration() const noexcept {
  return counterGeneration_;
}

template <int NumInput, int NumOutput, bool Bias, typename ParamConfig>
inline void EvolutionNet<NumInput, NumOutput, Bias, ParamConfig>::computeNetworks() {
  const std::size_t popSize = population_.getPopulationSize();
  assert(networks_.size() == popSize);

  for (std::size_t i = 0; i < popSize; ++i) {
    networks_[i].initializeFromGenome(population_.getGenomeNth(i));
  }
}

}  // namespace EvolutionNet

#endif  // EVOLUTION_NET_EVOLUTION_NET_HPP
