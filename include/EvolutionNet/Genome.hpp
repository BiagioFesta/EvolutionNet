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
#ifndef EVOLUTION_NET_GENOME_HPP
#define EVOLUTION_NET_GENOME_HPP

#include <EvolutionNet/ConnectionGene.hpp>
#include <EvolutionNet/FlatMap.hpp>
#include <EvolutionNet/FlatSet.hpp>
#include <EvolutionNet/Random.hpp>
#include <EvolutionNet/Types.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace EvolutionNet {

template <int NumInput, int NumOutput, bool Bias>
class Genome final {
 public:
  static constexpr int NumOutputValue = NumOutput;
  static constexpr bool IsBias = Bias;
  static constexpr int NumInputValue = NumInput;

  //! \brief Default constructor.
  Genome() = default;

  /*! \brief Initialize a new Genome.
       The new genome will be a minimal fully connected network.
   */
  template <typename ParamConfig>
  void initialize(RndEngine* rndEngine);

  /*! \brief Apply a random mutation to this genome.
   */
  template <typename ParamConfig>
  void mutate(RndEngine* rndEngine);

  /*! \brief Crossover between genomes.
   *  \note This assume `genome1` has better fitness.
   */
  template <typename ParamConfig>
  void crossover(const Genome& genome1,
                 const Genome& genome2,
                 RndEngine* rndEngine);

  //! \return the distance (0 means they are equivalents) among genomes.
  template <typename ParamConfig>
  float computeSimilarity(const Genome& oth) const noexcept;

  //! \return the number of connections.
  inline std::size_t getNumConnectionGenes() const noexcept;

  //! \return the number of nodes (it counts bias if present).
  inline std::size_t getNumNodes() const noexcept;

  //! \return the layer id of a specific internal node.
  inline LayerId getLayerOfNode(const NodeId nodeId) const noexcept;

  //! \return all layers IDs.
  inline const FlatMap<LayerId, FlatSet<NodeId>>& getLayers() const noexcept;

  //! \return all connection genes..
  inline const std::vector<ConnectionGene>& getConnections() const noexcept;

 private:
  NodeId nextNodeID_ = !Bias;

  FlatMap<ConnectionHash, ConnectionGene> connections_;

  FlatMap<LayerId, FlatSet<NodeId>> layers_;
  std::vector<LayerId> layerOfNodes_;

  template <typename ParamConfig>
  static void mutateWeight(RndEngine* rndEngine,
                           ConnectionGene* connectionGene);

  template <typename ParamConfig>
  void mutateAddNewNode(RndEngine* rndEngine);

  template <typename ParamConfig>
  void mutateAddNewConnection(RndEngine* rndEngine);

  bool validate() const;

  inline void emplaceNewConnectionGene(const NodeId from,
                                       const NodeId to,
                                       const float weight);

  inline void assignLayerNewNode(const NodeId nodeId, const LayerId layerId);

  inline ConnectionGene* existConnection(const NodeId from,
                                         const NodeId to) noexcept;

  static inline ConnectionHash hashConnection(const NodeId from,
                                              const NodeId to) noexcept;
};

template <int NumInput, int NumOutput, bool Bias>
template <typename ParamConfig>
void Genome<NumInput, NumOutput, Bias>::initialize(RndEngine* rndEngine) {
  constexpr int totalNumConnection = (NumInput + Bias) * NumOutput;
  constexpr NodeId nodeLastIndex = NumInput + NumOutput + 1;
  constexpr LayerId firstLayer = 0;
  constexpr LayerId lastLayer = std::numeric_limits<LayerId>::max() / 2;

  // Create connection (fully connected network)
  connections_.clear();
  connections_.reserve(totalNumConnection);
  for (NodeId from = (Bias ? 0 : 1); from <= NumInput; ++from) {
    for (NodeId to = NumInput + 1; to < nodeLastIndex; ++to) {
      emplaceNewConnectionGene(from, to, ParamConfig::NewRndWeight(rndEngine));
    }
  }

  // Assign layers
  layers_.clear();
  layerOfNodes_.clear();
  layerOfNodes_.reserve(nodeLastIndex);

  for (NodeId nodeId = 0; nodeId < nodeLastIndex; ++nodeId) {
    const LayerId layer = nodeId <= NumInput ? firstLayer : lastLayer;
    assignLayerNewNode(nodeId, layer);
  }

  nextNodeID_ = nodeLastIndex;

  assert(validate());
}

template <int NumInput, int NumOutput, bool Bias>
template <typename ParamConfig>
void Genome<NumInput, NumOutput, Bias>::mutate(RndEngine* rndEngine) {
  // For all connections check mutation weights
  for (ConnectionGene& connectionGene : connections_.valuesVector()) {
    if (CheckProbability(rndEngine, ParamConfig::ProbMutationWeight)) {
      mutateWeight<ParamConfig>(rndEngine, &connectionGene);
    }
  }

  // Check creation of new node
  if (CheckProbability(rndEngine, ParamConfig::ProbMutatationNewNode)) {
    mutateAddNewNode<ParamConfig>(rndEngine);
  }

  // Check creation of new connection
  if (CheckProbability(rndEngine, ParamConfig::ProbMutationNewConnection)) {
    mutateAddNewConnection<ParamConfig>(rndEngine);
  }

  assert(validate());
}

template <int NumInput, int NumOutput, bool Bias>
template <typename ParamConfig>
void Genome<NumInput, NumOutput, Bias>::crossover(const Genome& genome1,
                                                  const Genome& genome2,
                                                  RndEngine* rndEngine) {
  connections_.clear();

  const ConnectionGene* inheritGene;
  ConnectionHash hashConn;

  connections_.reserve(genome1.connections_.size());
  for (const ConnectionGene& gene1 : genome1.connections_.valuesVector()) {
    inheritGene = &gene1;
    hashConn = hashConnection(gene1.getFrom(), gene1.getTo());

    if (const ConnectionGene* gene2 = genome2.connections_[hashConn];
        gene2 &&
        !CheckProbability(rndEngine, ParamConfig::ProbInheritOnFitterGenome)) {
      inheritGene = gene2;
    }

    ConnectionGene* newConn = connections_.insert(hashConn, *inheritGene);
    if (!CheckProbability(rndEngine, ParamConfig::ProbInheritDisabledGene)) {
      newConn->setEnabled(true);
    }
  }

  layers_ = genome1.layers_;
  layerOfNodes_ = genome1.layerOfNodes_;

  nextNodeID_ = genome1.nextNodeID_;

  assert(validate());
}

template <int NumInput, int NumOutput, bool Bias>
template <typename ParamConfig>
float Genome<NumInput, NumOutput, Bias>::computeSimilarity(
    const Genome& oth) const noexcept {
  int N;
  int numExcess = 0, numDisjoint = 0, numMatching = 0;
  float weightDifference = 0.f;

  std::size_t i = 0, j = 0;
  while (i != connections_.size() || j != oth.connections_.size()) {
    if (i == connections_.size()) {
      ++numExcess;
      ++j;
    } else if (j == oth.connections_.size()) {
      ++numExcess;
      ++i;
    } else if (ConnectionHash hashI = connections_.nthKey(i),
               hashJ = oth.connections_.nthKey(j);
               hashI == hashJ) {
      ++numMatching;
      weightDifference += std::abs(connections_.nthValue(i).getWeight() -
                                   oth.connections_.nthValue(j).getWeight());
      ++i;
      ++j;
    } else if (connections_.nthKey(i) < oth.connections_.nthKey(j)) {
      ++numDisjoint;
      ++i;
    } else {
      ++numDisjoint;
      ++j;
    }
  }

  weightDifference = numMatching ? weightDifference / numMatching : 0.f;

  N = std::max(getNumConnectionGenes(), oth.getNumConnectionGenes());
  N = N < ParamConfig::NormalizedSizeGene ? 1 : N;

  const float t1 = ParamConfig::SimilarityCoefExcess *
                   static_cast<float>(numExcess) / static_cast<float>(N);
  const float t2 = ParamConfig::SimilarityCoefDisj *
                   static_cast<float>(numDisjoint) / static_cast<float>(N);
  const float t3 = ParamConfig::SimilarityCoefWeight * weightDifference;

  return t1 + t2 + t3;
}

template <int NumInput, int NumOutput, bool Bias>
inline std::size_t Genome<NumInput, NumOutput, Bias>::getNumConnectionGenes()
    const noexcept {
  return connections_.size();
}

template <int NumInput, int NumOutput, bool Bias>
inline std::size_t Genome<NumInput, NumOutput, Bias>::getNumNodes()
    const noexcept {
  return static_cast<std::size_t>(nextNodeID_ - !Bias);
}

template <int NumInput, int NumOutput, bool Bias>
inline LayerId Genome<NumInput, NumOutput, Bias>::getLayerOfNode(
    const NodeId nodeId) const noexcept {
  assert(static_cast<std::size_t>(nodeId) < layerOfNodes_.size());
  assert(Bias || nodeId != 0);
  return layerOfNodes_[nodeId];
}

template <int NumInput, int NumOutput, bool Bias>
inline const FlatMap<LayerId, FlatSet<NodeId>>&
Genome<NumInput, NumOutput, Bias>::getLayers() const noexcept {
  return layers_;
}

template <int NumInput, int NumOutput, bool Bias>
inline const std::vector<ConnectionGene>&
Genome<NumInput, NumOutput, Bias>::getConnections() const noexcept {
  return connections_.valuesVector();
}

template <int NumInput, int NumOutput, bool Bias>
template <typename ParamConfig>
void Genome<NumInput, NumOutput, Bias>::mutateWeight(
    RndEngine* rndEngine,
    ConnectionGene* connectionGene) {
  const float newWeight =
      CheckProbability(rndEngine, ParamConfig::ProbMutationWeightPerturbation)
          ? ParamConfig::WeightPerturbation(connectionGene->getWeight(),
                                            rndEngine)
          : ParamConfig::NewRndWeight(rndEngine);

  connectionGene->setWeight(newWeight);
  assert(newWeight >= -1.f && newWeight <= 1.f);
}

template <int NumInput, int NumOutput, bool Bias>
template <typename ParamConfig>
void Genome<NumInput, NumOutput, Bias>::mutateAddNewNode(RndEngine* rndEngine) {
  // Pick random connection
  const std::size_t index = std::uniform_int_distribution<std::size_t>{
      static_cast<std::size_t>(0), connections_.size() - 1}(*rndEngine);
  ConnectionGene& oldConnection = connections_.nthValue(index);
  const NodeId prevNode = oldConnection.getFrom();
  const NodeId succNode = oldConnection.getTo();
  const float weight = oldConnection.getWeight();

  // TODO(bfesta): if connection not enable we skip. is that correct?
  if ((Bias && prevNode == 0) || !oldConnection.getEnabled()) {
    // TODO(bfesta): this is the fastest way. However, slower way in order to
    // guarantee the adding.
    return;
  }

  // Disable old connection
  oldConnection.setEnabled(false);

  // Create two new connections (note: oldConnection is invalid reference now
  // because of emplace)
  emplaceNewConnectionGene(prevNode, nextNodeID_, 1.f);
  emplaceNewConnectionGene(nextNodeID_, succNode, weight);

  // Create bias connection
  if constexpr (Bias) {
    emplaceNewConnectionGene(0, nextNodeID_, 0.f);
  }

  // Assign the layer
  const LayerId layer =
      (getLayerOfNode(prevNode) + getLayerOfNode(succNode)) / 2;
  assignLayerNewNode(nextNodeID_, layer);

  // Increment Next Node ID counter
  ++nextNodeID_;
}

template <int NumInput, int NumOutput, bool Bias>
template <typename ParamConfig>
void Genome<NumInput, NumOutput, Bias>::mutateAddNewConnection(
    RndEngine* rndEngine) {
  std::uniform_int_distribution<std::size_t> rndLayer{
      static_cast<std::size_t>(0), layers_.size() - 1};
  std::size_t layer1, layer2;

  do {
    layer1 = rndLayer(*rndEngine);
    layer2 = rndLayer(*rndEngine);
  } while (layer1 == layer2);

  if (layer2 < layer1) {
    std::swap(layer1, layer2);
  }

  const FlatSet<NodeId>& layer1_data = layers_.nthValue(layer1);
  const FlatSet<NodeId>& layer2_data = layers_.nthValue(layer2);

  std::uniform_int_distribution<std::size_t> rndFrom{
      static_cast<std::size_t>(0), layer1_data.size() - 1};
  std::uniform_int_distribution<std::size_t> rndTo{static_cast<std::size_t>(0),
                                                   layer2_data.size() - 1};

  const NodeId from = layer1_data.nth(rndFrom(*rndEngine));
  const NodeId to = layer2_data.nth(rndTo(*rndEngine));

  // Check connection already exists
  if (auto connection = existConnection(from, to); connection != nullptr) {
    connection->setEnabled(true);
  } else {
    emplaceNewConnectionGene(from, to, ParamConfig::NewRndWeight(rndEngine));
  }
}

template <int NumInput, int NumOutput, bool Bias>
bool Genome<NumInput, NumOutput, Bias>::validate() const {
  if (layerOfNodes_.size() != nextNodeID_) {
    assert(false);
    return false;
  }

  for (const auto& connectionGene : connections_.valuesVector()) {
    if (layerOfNodes_.size() <= connectionGene.getFrom()) {
      assert(false);
      return false;
    }
    if (layerOfNodes_.size() <= connectionGene.getTo()) {
      assert(false);
      return false;
    }
    if (layerOfNodes_[connectionGene.getTo()] <=
        layerOfNodes_[connectionGene.getFrom()]) {
      assert(false);
      return false;
    }
  }

  return true;
}

template <int NumInput, int NumOutput, bool Bias>
inline void Genome<NumInput, NumOutput, Bias>::emplaceNewConnectionGene(
    const NodeId from,
    const NodeId to,
    const float weight) {
  const ConnectionHash connectionHash = hashConnection(from, to);
  assert(existConnection(from, to) == nullptr);

  connections_.insert(connectionHash, from, to, weight, true);
}

template <int NumInput, int NumOutput, bool Bias>
inline void Genome<NumInput, NumOutput, Bias>::assignLayerNewNode(
    const NodeId node,
    const LayerId layer) {
  assert(node == static_cast<NodeId>(layerOfNodes_.size()));

  if (auto* layerNodes = layers_[layer]; layerNodes != nullptr) {
    assert(std::find(layerNodes->begin(), layerNodes->end(), node) ==
           layerNodes->end());
    layerNodes->insert(node);
  } else {
    layers_.insert(layer)->insert(node);
  }
  layerOfNodes_.emplace_back(layer);
}

template <int NumInput, int NumOutput, bool Bias>
inline ConnectionGene* Genome<NumInput, NumOutput, Bias>::existConnection(
    const NodeId from,
    const NodeId to) noexcept {
  return connections_[hashConnection(from, to)];
}

template <int NumInput, int NumOutput, bool Bias>
inline ConnectionHash Genome<NumInput, NumOutput, Bias>::hashConnection(
    const NodeId from,
    const NodeId to) noexcept {
  ConnectionHash hash = 0;
  hash |= static_cast<ConnectionHash>(from);
  hash |= static_cast<ConnectionHash>(to) << 32;
  return hash;
}

}  // namespace EvolutionNet

#endif  // EVOLUTION_NET_GENOME_HPP
