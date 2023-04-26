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
#ifndef EVOLUTION_NET_POPULATION_HPP
#define EVOLUTION_NET_POPULATION_HPP

#include <EvolutionNet/Random.hpp>
#include <EvolutionNet/Types.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <utility>
#include <vector>

namespace EvolutionNet {

template <typename Genome>
class Population final {
 public:
  //! \brief Default constructor.
  Population() = default;

  //! \brief Initialize a population of genomes.
  template <typename ParamConfig>
  void initialize(const std::size_t size, RndEngine* rndEngine);

  //! \brief Assign a fitness to the genomeIndex-th member of the population.
  inline void assignFitness(const std::size_t genomeIndex,
                            const FitnessScore fitness) noexcept;

  //! \brief Evolve the population to the next generation.
  template <typename ParamConfig>
  void nextGeneration(RndEngine* rndEngine);

  //! \return The size of population.
  inline std::size_t getPopulationSize() const noexcept;

  //! \return the n-th genome of the population.
  inline const Genome& getGenomeNth(const std::size_t index) const noexcept;

 private:
  using Specie = std::vector<std::size_t>;

  std::vector<Genome> population_;
  std::vector<FitnessScore> fitness_;
  std::vector<FitnessScore> adjustedFitness_;
  std::vector<std::size_t> specieOfGenomes_;

  std::vector<const Genome*> specieRapresentatives_;
  std::vector<Specie> species_;
  std::vector<FitnessScore> specieFitnessSum_;
  std::vector<FitnessScore> specieFitnessMax_;
  std::vector<std::size_t> specieStagnantGenerations_;
  std::vector<std::size_t> specieBounds_;

  template <typename ParamConfig>
  void speciate();

  void computeAdjustedFitness();

  void sortSpecies();

  void computeSpeciesBounds();

  template <typename ParamConfig>
  void offspringSpecies(RndEngine* rndEngine);

  void cleanSpecies();

  void electRapresentativeForSpecies(RndEngine* rndEngine);

  template <typename ParamConfig>
  void computeStagnation();

  inline FitnessScore computeMaxFitnessInSpecie(
      const std::size_t specieIndex) const noexcept;
};

template <typename Genome>
template <typename ParamConfig>
void Population<Genome>::initialize(const std::size_t size,
                                    RndEngine* rndEngine) {
  population_.clear();
  population_.resize(size);
  for (std::size_t i = 0; i < size; ++i) {
    population_[i].template initialize<ParamConfig>(rndEngine);
  }

  fitness_.resize(size);
  adjustedFitness_.resize(size);
  specieOfGenomes_.resize(size);

  specieRapresentatives_.clear();
  species_.clear();
  specieFitnessSum_.clear();
  specieFitnessMax_.clear();
  specieStagnantGenerations_.clear();
}

template <typename Genome>
inline void Population<Genome>::assignFitness(
    const std::size_t genomeIndex,
    const FitnessScore fitness) noexcept {
  assert(genomeIndex < fitness_.size());
  fitness_[genomeIndex] = fitness;
}

template <typename Genome>
template <typename ParamConfig>
void Population<Genome>::nextGeneration(RndEngine* rndEngine) {
  speciate<ParamConfig>();
  computeAdjustedFitness();
  sortSpecies();
  computeStagnation<ParamConfig>();
  computeSpeciesBounds();
  offspringSpecies<ParamConfig>(rndEngine);
  electRapresentativeForSpecies(rndEngine);
  cleanSpecies();
}

template <typename Genome>
inline std::size_t Population<Genome>::getPopulationSize() const noexcept {
  return population_.size();
}

template <typename Genome>
inline const Genome& Population<Genome>::getGenomeNth(
    const std::size_t index) const noexcept {
  assert(index < population_.size());
  return population_[index];
}

template <typename Genome>
template <typename ParamConfig>
void Population<Genome>::speciate() {
  specieFitnessSum_.clear();
  specieFitnessSum_.resize(species_.size(), 0.f);

  for (auto& specie : species_) {
    specie.clear();
  }

  assert(specieRapresentatives_.size() == species_.size());
  assert(specieOfGenomes_.size() == population_.size());

  // Assign each genome to a specie
  bool found;
  for (std::size_t genomeIndex = 0; genomeIndex < population_.size();
       ++genomeIndex) {
    const auto& genome = population_[genomeIndex];
    found = false;

    for (std::size_t specieIndex = 0;
         specieIndex < specieRapresentatives_.size() && !found;
         ++specieIndex) {
      const Genome* representative = specieRapresentatives_[specieIndex];
      if (genome.template computeSimilarity<ParamConfig>(*representative) <
          ParamConfig::SpeciesSimilarityThreshold) {
        species_[specieIndex].push_back(genomeIndex);
        specieOfGenomes_[genomeIndex] = specieIndex;
        found = true;
      }
    }

    if (!found) {
      specieRapresentatives_.push_back(&genome);
      specieOfGenomes_[genomeIndex] = species_.size();
      species_.emplace_back(1, genomeIndex);
      specieFitnessSum_.push_back(0.f);
      specieFitnessMax_.push_back(0.f);
      specieStagnantGenerations_.push_back(0);
    }
  }
}

template <typename Genome>
void Population<Genome>::computeAdjustedFitness() {
  assert(fitness_.size() == population_.size());
  assert(specieOfGenomes_.size() == population_.size());
  assert(specieFitnessSum_.size() == species_.size());

  adjustedFitness_.resize(fitness_.size());

  for (std::size_t genomeIndex = 0; genomeIndex < population_.size();
       ++genomeIndex) {
    const std::size_t specieIndex = specieOfGenomes_[genomeIndex];
    assert(specieIndex < species_.size());
    const float adjustedFitness =
        fitness_[genomeIndex] / species_[specieIndex].size();

    adjustedFitness_[genomeIndex] = adjustedFitness;
    specieFitnessSum_[specieIndex] += adjustedFitness;
  }
}

template <typename Genome>
void Population<Genome>::sortSpecies() {
  for (auto& specie : species_) {
    std::sort(specie.begin(),
              specie.end(),
              [this](const std::size_t i, const std::size_t j) noexcept {
                return adjustedFitness_[j] < adjustedFitness_[i];
              });
  }
}

template <typename Genome>
void Population<Genome>::computeSpeciesBounds() {
  assert(specieFitnessSum_.size() == species_.size());
  const FitnessScore sumFitness = std::accumulate(specieFitnessSum_.cbegin(),
                                                  specieFitnessSum_.cend(),
                                                  static_cast<FitnessScore>(0));
  specieBounds_.resize(species_.size());

  for (std::size_t specieIndex = 0; specieIndex < species_.size();
       ++specieIndex) {
    const std::size_t bound = static_cast<size_t>(
        std::ceil(specieFitnessSum_[specieIndex] / sumFitness *
                  species_[specieIndex].size()));
    assert(bound <= species_[specieIndex].size());
    specieBounds_[specieIndex] = bound;
  }
}

template <typename Genome>
template <typename ParamConfig>
void Population<Genome>::offspringSpecies(RndEngine* rndEngine) {
  assert(species_.size() == specieBounds_.size());
  const Genome *parent1, *parent2;

  if (std::all_of(
          species_.cbegin(),
          species_.cend(),
          [](const Specie& specie) noexcept { return specie.empty(); })) {
    const std::size_t bound = population_.size() / 2;
    for (std::size_t i = 0; i < bound; ++i) {
      population_[i].template mutate<ParamConfig>(rndEngine);
    }
    for (std::size_t i = bound; i < population_.size(); ++i) {
      population_[i].template initialize<ParamConfig>(rndEngine);
    }
    return;
  }

  for (std::size_t specieIndex = 0; specieIndex < species_.size();
       ++specieIndex) {
    const Specie& specie = species_[specieIndex];
    if (specie.empty()) {
      continue;
    }
    const std::size_t bound = specieBounds_[specieIndex];
    assert(bound > 0);

    // old offspring (to keep)
    for (std::size_t i =
             (ParamConfig::SizeSpecieForChampion < specie.size() ? 1 : 0);
         i < bound;
         ++i) {
      Genome& genome = population_[specie[i]];
      genome.template mutate<ParamConfig>(rndEngine);
    }

    // new offstring (to create with mating)
    for (std::size_t i = bound; i < specie.size(); ++i) {
      Genome& genome = population_[specie[i]];

      if (CheckProbability(rndEngine, ParamConfig::ProbOffspringCrossover)) {
        if (CheckProbability(rndEngine, ParamConfig::ProbMatingInterspecies)) {
          // Interspecie mating
          std::uniform_int_distribution<std::size_t> rndOthSpecie{
              0, species_.size() - 1};
          std::size_t othSpecieIndex;
          do {
            othSpecieIndex = rndOthSpecie(*rndEngine);
          } while (species_[othSpecieIndex].empty() ||
                   specieBounds_[othSpecieIndex] == 0);
          const Specie& othSpecie = species_[othSpecieIndex];
          const std::size_t bound2 = specieBounds_[othSpecieIndex];
          const std::size_t id1 = std::uniform_int_distribution<std::size_t>{
              0, bound - 1}(*rndEngine);
          const std::size_t id2 = std::uniform_int_distribution<std::size_t>{
              0, bound2 - 1}(*rndEngine);
          if (adjustedFitness_[specie[id1]] <
              adjustedFitness_[othSpecie[id2]]) {
            parent1 = &(population_[othSpecie[id2]]);
            parent2 = &(population_[specie[id1]]);
          } else {
            parent1 = &(population_[specie[id1]]);
            parent2 = &(population_[othSpecie[id2]]);
          }
        } else {
          // Mating across same specie
          std::uniform_int_distribution<std::size_t> rndElm{0, bound - 1};
          const std::size_t id1 = rndElm(*rndEngine);
          const std::size_t id2 = rndElm(*rndEngine);
          if (id2 < id1) {
            parent1 = &(population_[specie[id2]]);
            parent2 = &(population_[specie[id1]]);
          } else {
            parent1 = &(population_[specie[id1]]);
            parent2 = &(population_[specie[id2]]);
          }
        }
        genome.template crossover<ParamConfig>(*parent1, *parent2, rndEngine);
      }  // if crossover

      genome.template mutate<ParamConfig>(rndEngine);
    }
  }
}

template <typename Genome>
void Population<Genome>::cleanSpecies() {
  assert(specieStagnantGenerations_.size() == species_.size());
  assert(specieFitnessMax_.size() == species_.size());

  // Clean all empty species
  std::vector<Specie> newSpecies;
  newSpecies.reserve(species_.size());
  std::vector<FitnessScore> newSpecieFitnessMax;
  newSpecies.reserve(species_.size());
  std::vector<std::size_t> newSpecieStagnantGenerations;
  newSpecies.reserve(species_.size());
  std::vector<const Genome*> newSpecieRapresentatives;
  newSpecieRapresentatives.reserve(species_.size());

  for (std::size_t i = 0; i < species_.size(); ++i) {
    if ((!species_[i].empty()) && (specieBounds_[i] > 0)) {
      newSpecies.push_back(std::move(species_[i]));
      newSpecieFitnessMax.push_back(std::move(specieFitnessMax_[i]));
      newSpecieStagnantGenerations.push_back(
          std::move(specieStagnantGenerations_[i]));
      newSpecieRapresentatives.push_back(std::move(specieRapresentatives_[i]));
    }
  }
  species_ = std::move(newSpecies);
  specieFitnessMax_ = std::move(newSpecieFitnessMax);
  specieStagnantGenerations_ = std::move(newSpecieStagnantGenerations);
  specieRapresentatives_ = std::move(newSpecieRapresentatives);
}

template <typename Genome>
void Population<Genome>::electRapresentativeForSpecies(RndEngine* rndEngine) {
  specieRapresentatives_.resize(species_.size());

  for (std::size_t i = 0; i < species_.size(); ++i) {
    const Specie& specie = species_[i];
    const std::size_t numToKeep = specieBounds_[i];
    if (!specie.empty() && numToKeep > 0) {
      const std::size_t rndElem = std::uniform_int_distribution<std::size_t>{
          0, numToKeep - 1}(*rndEngine);
      specieRapresentatives_[i] = &(population_[specie[rndElem]]);
    }
  }
}

template <typename Genome>
template <typename ParamConfig>
void Population<Genome>::computeStagnation() {
  assert(specieFitnessMax_.size() == species_.size());
  assert(specieStagnantGenerations_.size() == species_.size());

  for (std::size_t specieIndex = 0; specieIndex < species_.size();
       ++specieIndex) {
    if (!species_[specieIndex].empty()) {
      const FitnessScore currentMaxFitness =
          computeMaxFitnessInSpecie(specieIndex);
      const FitnessScore historicalMaxFitness = specieFitnessMax_[specieIndex];

      if (specieStagnantGenerations_[specieIndex] == 0 ||
          historicalMaxFitness < currentMaxFitness) {
        specieFitnessMax_[specieIndex] = currentMaxFitness;
        specieStagnantGenerations_[specieIndex] = 1;
      } else {
        if (ParamConfig::GenForStagningSpecies <
            (++specieStagnantGenerations_[specieIndex])) {
          species_[specieIndex].clear();
        }
      }
    }
  }
}

template <typename Genome>
inline FitnessScore Population<Genome>::computeMaxFitnessInSpecie(
    const std::size_t specieIndex) const noexcept {
  assert(specieIndex < species_.size());
  assert(!species_[specieIndex].empty());
  assert(species_[specieIndex][0] < adjustedFitness_.size());

  return adjustedFitness_[species_[specieIndex][0]];
}

}  // namespace EvolutionNet

#endif  // EVOLUTION_NET_POPULATION_HPP
