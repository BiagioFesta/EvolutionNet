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
#ifndef EVOLUTION_NET_FLAT_SET_HPP
#define EVOLUTION_NET_FLAT_SET_HPP
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <vector>

namespace EvolutionNet {

template <typename T>
class FlatSet {
 public:
  static_assert(std::is_integral_v<T>);
  using container_type = std::vector<T>;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  //! \brief Clear the set
  inline void clear() noexcept;

  //! \brief Reserve `size` elements (capacity)
  inline void reserve(const std::size_t size);

  //! \return the size of the container
  inline std::size_t size() const noexcept;

  //! \return the n-th element (they are kept sorted inside the container)
  inline T nth(const std::size_t nth) const noexcept;

  //! \return the begin iterator
  inline const_iterator begin() const noexcept;

  //! \return the end iterator
  inline const_iterator end() const noexcept;

  //! \return whether the container is empty or not
  inline bool empty() const noexcept;

  /*! \brief Insert a new element in the container.
   *  The new element will be inserted in a "sorted way" (the container will keep elements sorted)
   *  \note The element must not already exist in the container. Otherwise undefined behavior!
   */
  inline void insert(T value);

 private:
  std::vector<T> data_;
};

template <typename T>
inline void FlatSet<T>::clear() noexcept {
  data_.clear();
}

template <typename T>
inline void FlatSet<T>::reserve(const std::size_t size) {
  data_.reserve(size);
}

template <typename T>
inline std::size_t FlatSet<T>::size() const noexcept {
  return data_.size();
}

template <typename T>
inline T FlatSet<T>::nth(const std::size_t nth) const noexcept {
  assert(nth < data_.size());
  return data_[nth];
}

template <typename T>
inline typename FlatSet<T>::const_iterator FlatSet<T>::begin() const noexcept {
  return data_.cbegin();
}

template <typename T>
inline typename FlatSet<T>::const_iterator FlatSet<T>::end() const noexcept {
  return data_.cend();
}

template <typename T>
inline bool FlatSet<T>::empty() const noexcept {
  return data_.empty();
}

template <typename T>
inline void FlatSet<T>::insert(T value) {
  data_.push_back(value);

  T* beg = data_.data();
  T* end = beg + data_.size();
  T* last = end - 1;
  T* bound = std::lower_bound(beg, last, value);
  if (bound != last) {
    assert(*bound != value);
    std::rotate(bound, last, end);
  }
  assert(std::is_sorted(data_.cbegin(), data_.cend()));
}

}  // namespace EvolutionNet

#endif  // EVOLUTION_NET_FLAT_SET_HPP
