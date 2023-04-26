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
#ifndef EVOLUTION_NET_CONNECTION_GENE_HPP
#define EVOLUTION_NET_CONNECTION_GENE_HPP

#include <EvolutionNet/Types.hpp>

namespace EvolutionNet {

class ConnectionGene final {
 public:
  //! \brief Default copy constructor.
  ConnectionGene(const ConnectionGene&) = default;

  //! \brief Construct the aggregation.
  inline ConnectionGene(const NodeId from,
                        const NodeId to,
                        const float weight,
                        const bool enabled) noexcept;

  //! \return the weight of the link.
  inline float getWeight() const noexcept;

  //! \brief set the weight of the link.
  inline void setWeight(const float weight) noexcept;

  // !\return the Node ID of the outgoing connection.
  inline NodeId getFrom() const noexcept;

  //! \return the NodeID of the ingoing connection.
  inline NodeId getTo() const noexcept;

  // !\brief set whether the connection is enabled or not.
  inline void setEnabled(bool enabled) noexcept;

  //! \return whether the connection is enabled.
  inline bool getEnabled() const noexcept;

 private:
  NodeId from_;
  NodeId to_;
  float weight_;
  bool enabled_;
};

inline ConnectionGene::ConnectionGene(const NodeId from,
                                      const NodeId to,
                                      const float weight,
                                      const bool enabled) noexcept
    : from_{from}, to_{to}, weight_{weight}, enabled_{enabled} {}

inline float ConnectionGene::getWeight() const noexcept {
  return weight_;
}

inline void ConnectionGene::setWeight(const float weight) noexcept {
  weight_ = weight;
}

inline NodeId ConnectionGene::getFrom() const noexcept {
  return from_;
}

inline NodeId ConnectionGene::getTo() const noexcept {
  return to_;
}

inline void ConnectionGene::setEnabled(bool enabled) noexcept {
  enabled_ = enabled;
}

inline bool ConnectionGene::getEnabled() const noexcept {
  return enabled_;
}

}  // namespace EvolutionNet

#endif  // EVOLUTION_NET_CONNECTION_GENE_HPP
