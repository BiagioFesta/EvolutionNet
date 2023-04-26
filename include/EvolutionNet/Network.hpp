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
#ifndef EVOLUTION_NET_NETWORK_HPP
#define EVOLUTION_NET_NETWORK_HPP

#include <EvolutionNet/Types.hpp>
#include <algorithm>
#include <cassert>
#include <istream>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

namespace EvolutionNet {

template <typename Genome>
class Network final {
 public:
  //! \brief Inizialize a network from a genome.
  void initializeFromGenome(const Genome& genome);

  //! \brief Set the input value (input-th).
  inline void setInputValue(const std::size_t input,
                            const float value) noexcept;

  //! \return the output value (output-th). Remember to `feedforward` before
  //! take this value.
  inline float getOutputValue(const std::size_t output) const noexcept;

  //! \brief simply, feed forwars algorithm network.
  template <typename ParamConfig>
  void feedForward();

  //! \brief assign a fitness value to this network. Higher value means better
  //! score.
  inline void setFitness(const FitnessScore fitness) noexcept;

  //! \return the fitness score of this network.
  inline FitnessScore getFitness() const noexcept;

  /*! \brief Serialize network structure.
   *         Only the network structure will be serialized.
   *         FitnessScore and Node values will NOT be serialized.
   */
  void serialize(std::ostream* os, const std::size_t bufferSize = 1024) const;

  /*! \brief Deserialize network structure.
   *  \see `serialize(std::ostream*, const std::size_t bufferSize)` method.
   *  \return false in case of error.
   */
  bool deserialize(std::istream* is);

  /*! \brief Check equality between two network structures.
   *  \note Only structure will be checked. Fitness value and current node
   * values will not be checked.
   */
  bool sameStructure(const Network& rhs) const noexcept;

 private:
  static constexpr std::uint32_t MagicNumber = 0xbf0ebf03;

  using IncomingConnection = std::pair<NodeId, float>;

  std::vector<NodeId> nodes_;
  std::vector<float> nodeValues_;
  std::vector<std::vector<IncomingConnection>> inConnections_;
  FitnessScore fitness_ = 0.f;
};

template <typename Genome>
void Network<Genome>::initializeFromGenome(const Genome& genome) {
  const std::size_t numNodes = genome.getNumNodes() + !Genome::IsBias;

  nodes_.clear();
  nodes_.reserve(numNodes);

  const auto& layers = genome.getLayers();
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const auto& nodes = layers.nthValue(i);
    for (const NodeId nodeId : nodes) {
      nodes_.push_back(nodeId);
    }
  }

  nodeValues_.clear();
  nodeValues_.resize(numNodes);

  inConnections_.clear();
  inConnections_.resize(numNodes);

  for (const auto& connGene : genome.getConnections()) {
    assert(connGene.getTo() < inConnections_.size());
    if (connGene.getEnabled()) {
      inConnections_[connGene.getTo()].emplace_back(connGene.getFrom(),
                                                    connGene.getWeight());
    }
  }

  fitness_ = 0.f;
}

template <typename Genome>
inline void Network<Genome>::setInputValue(const std::size_t input,
                                           const float value) noexcept {
  assert(input < Genome::NumInputValue);
  assert(input + 1 < nodeValues_.size());
  nodeValues_[input + 1] = value;
}

template <typename Genome>
inline float Network<Genome>::getOutputValue(
    const std::size_t output) const noexcept {
  assert(output < Genome::NumOutputValue);
  assert(Genome::NumInputValue + 1 + output < nodeValues_.size());
  return nodeValues_[Genome::NumInputValue + 1 + output];
}

template <typename Genome>
template <typename ParamConfig>
void Network<Genome>::feedForward() {
  assert(!nodes_.empty());
  nodeValues_[0] = Genome::IsBias ? 1.f : 0.f;

  for (std::size_t i = Genome::NumInputValue + 1; i < nodes_.size(); ++i) {
    const NodeId nodeId = nodes_[i];
    assert(nodeId < inConnections_.size());
    assert(nodeId < nodeValues_.size());
    assert(!inConnections_[nodeId].empty());

    nodeValues_[nodeId] = 0.f;
    for (const auto& inConn : inConnections_[nodeId]) {
      nodeValues_[nodeId] += nodeValues_[inConn.first] * inConn.second;
    }
    nodeValues_[nodeId] = ParamConfig::ActivationFn(nodeValues_[nodeId]);
  }
}

template <typename Genome>
inline void Network<Genome>::setFitness(const FitnessScore fitness) noexcept {
  fitness_ = fitness;
}

template <typename Genome>
inline FitnessScore Network<Genome>::getFitness() const noexcept {
  return fitness_;
}

template <typename Genome>
void Network<Genome>::serialize(std::ostream* os,
                                const std::size_t bufferSize) const {
  const std::size_t TotalNumConnections =
      std::accumulate(inConnections_.cbegin(),
                      inConnections_.cend(),
                      std::size_t{0},
                      [](const auto& acc, const auto& inConnections) noexcept {
                        return acc + inConnections.size();
                      });
  const std::size_t TotalSize =
      (4)                                 // Magic number
      + (4)                               // Num of nodes
      + (4 * nodes_.size())               // Node IDs
      + (4 * nodes_.size())               // Num In Connections for each node
      + ((4 + 4) * TotalNumConnections);  // Connection Data

  std::size_t offset = 0, bytesToWrite;
  auto data = std::make_unique<unsigned char[]>(TotalSize);

  // Magic Number (4 bytes)
  assert(offset + 4 <= TotalSize);
  *reinterpret_cast<std::uint32_t*>(data.get()) = MagicNumber;
  offset += 4;

  // Num of Nodes (4 bytes)
  assert(offset + 4 <= TotalSize);
  *reinterpret_cast<std::uint32_t*>(data.get() + 4) =
      static_cast<std::uint32_t>(nodes_.size());
  offset += 4;

  for (const NodeId nodeId : nodes_) {
    assert(offset + 4 <= TotalSize);
    *reinterpret_cast<std::uint32_t*>(data.get() + offset) = nodeId;
    offset += 4;
  }

  for (const auto& inConnections : inConnections_) {
    assert(offset + 4 <= TotalSize);
    *reinterpret_cast<std::uint32_t*>(data.get() + offset) =
        static_cast<std::uint32_t>(inConnections.size());
    offset += 4;

    for (const auto& [nodeFrom, weight] : inConnections) {
      assert(offset + 4 <= TotalSize);
      *reinterpret_cast<std::uint32_t*>(data.get() + offset) = nodeFrom;
      offset += 4;

      assert(offset + 4 <= TotalSize);
      static_assert(sizeof(float) == sizeof(std::uint32_t));
      *reinterpret_cast<std::uint32_t*>(data.get() + offset) =
          *reinterpret_cast<const std::uint32_t*>(&weight);
      offset += 4;
    }
  }

  assert(offset == TotalSize);

  offset = 0;
  while (offset < TotalSize) {
    bytesToWrite = std::min(TotalSize - offset, bufferSize);
    os->write(reinterpret_cast<const char*>(data.get() + offset), bytesToWrite);
    offset += bytesToWrite;
  }

  assert(offset == TotalSize);
}

template <typename Genome>
bool Network<Genome>::deserialize(std::istream* is) {
  std::uint32_t buffer4, numNodes, numInConnections;
  std::uint64_t buffer8;

  nodes_.clear();
  nodeValues_.clear();
  inConnections_.clear();
  fitness_ = 0.f;

  is->read(reinterpret_cast<char*>(&buffer4), 4);
  if (is->gcount() != 4 || buffer4 != MagicNumber) {
    return false;
  }

  is->read(reinterpret_cast<char*>(&buffer4), 4);
  if (is->gcount() != 4) {
    return false;
  }
  numNodes = buffer4;

  nodes_.reserve(numNodes);
  nodeValues_.resize(numNodes);
  inConnections_.resize(numNodes);
  for (NodeId i = 0; i < numNodes; ++i) {
    is->read(reinterpret_cast<char*>(&buffer4), 4);
    if (is->gcount() != 4) {
      return false;
    }

    nodes_.push_back(buffer4);
  }

  for (NodeId i = 0; i < numNodes; ++i) {
    is->read(reinterpret_cast<char*>(&buffer4), 4);
    if (is->gcount() != 4) {
      return false;
    }

    numInConnections = buffer4;
    inConnections_[i].reserve(numInConnections);
    for (std::uint32_t j = 0; j < numInConnections; ++j) {
      is->read(reinterpret_cast<char*>(&buffer8), 8);
      if (is->gcount() != 8) {
        return false;
      }

      inConnections_[i].emplace_back(
          *reinterpret_cast<NodeId*>(&buffer8),
          *reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(&buffer8) +
                                    sizeof(NodeId)));
    }
  }

  return true;
}

template <typename Genome>
bool Network<Genome>::sameStructure(const Network& rhs) const noexcept {
  if (nodes_ != rhs.nodes_) {
    return false;
  }

  assert(inConnections_.size() == rhs.inConnections_.size());
  for (std::size_t i = 0; i < inConnections_.size(); ++i) {
    if (inConnections_[i] != rhs.inConnections_[i]) {
      return false;
    }
  }

  return true;
}

}  // namespace EvolutionNet

#endif  // EVOLUTION_NET_NETWORK_HPP
