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
#ifndef EVOLUTION_NET_FLAT_MAP_HPP
#define EVOLUTION_NET_FLAT_MAP_HPP
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

namespace EvolutionNet {

template <typename Key, typename Value>
class FlatMap {
 public:
  static_assert(std::is_integral_v<Key>);
  using KeyIndex = std::pair<Key, std::size_t>;
  using KeyIndices = std::vector<KeyIndex>;

  /*! \brief Insert a *new* element in the map.
   *  \return a pointer to the new inserted value element.
   *  \note The key must not exist. Otherwise the structure is broken.
   *  \note Complexity O(logN)
   *  \note It might invalidate all existing references.
   */
  template <typename... Args>
  inline Value* insert(const Key key, Args&&... args) noexcept;

  /*! \brief Scan for a key and return the associated element.
   *  \return the element. If key not found, then it returns `nullptr`.
   */
  inline const Value* operator[](const Key key) const noexcept;
  inline Value* operator[](const Key key) noexcept;

  //! \brief Clean the data structure.
  inline void clear() noexcept;

  //! \brief Reserve for a number of elements.
  inline void reserve(const std::size_t size) noexcept;

  //! \return the number of elements contained in the data structure.
  inline std::size_t size() const noexcept;

  //! \return the n-th key (since they are sorted). UB if out-of-bound.
  inline Key nthKey(const std::size_t nth) const noexcept;

  //! \return the n-th value (since they are sorted). UB if out-of-bound.
  inline const Value& nthValue(const std::size_t nth) const noexcept;
  inline Value& nthValue(const std::size_t nth) noexcept;

  /*! \return the vector containing all values.
   *  \note The order of values is not sorted! They are sorted by order of insertion.
   */
  inline const std::vector<Value>& valuesVector() const noexcept;
  inline std::vector<Value>& valuesVector() noexcept;

 private:
  KeyIndices keys_;
  std::vector<Value> values_;
};

template <typename Key, typename Value>
template <typename... Args>
inline Value* FlatMap<Key, Value>::insert(const Key key, Args&&... args) noexcept {
  assert(this->operator[](key) == nullptr);

  keys_.emplace_back(key, values_.size());
  values_.emplace_back(std::forward<Args>(args)...);

  KeyIndex* beg = keys_.data();
  KeyIndex* end = beg + keys_.size();
  KeyIndex* last = end - 1;
  KeyIndex* bound =
      std::lower_bound(beg, last, key, [](const KeyIndex& el, const Key& key) noexcept { return el.first < key; });
  std::rotate(bound, last, end);

  assert(std::is_sorted(keys_.cbegin(), keys_.cend()));
  assert(std::adjacent_find(keys_.cbegin(), keys_.cend()) == keys_.cend());
  assert(keys_.size() == values_.size());

  return &values_.back();
}

template <typename Key, typename Value>
inline Value* FlatMap<Key, Value>::operator[](const Key key) noexcept {
  const auto finder = std::lower_bound(
      keys_.begin(), keys_.end(), key, [](const KeyIndex& el, const Key& key) noexcept { return el.first < key; });
  if (finder != keys_.end() && finder->first == key) {
    return &(values_[finder->second]);
  }
  return nullptr;
}

template <typename Key, typename Value>
inline const Value* FlatMap<Key, Value>::operator[](const Key key) const noexcept {
  const auto finder = std::lower_bound(
      keys_.cbegin(), keys_.cend(), key, [](const KeyIndex& el, const Key& key) noexcept { return el.first < key; });
  if (finder != keys_.cend() && finder->first == key) {
    return &(values_[finder->second]);
  }
  return nullptr;
}

template <typename Key, typename Value>
inline void FlatMap<Key, Value>::clear() noexcept {
  keys_.clear();
  values_.clear();
}

template <typename Key, typename Value>
inline void FlatMap<Key, Value>::reserve(const std::size_t size) noexcept {
  keys_.reserve(size);
  values_.reserve(size);
}

template <typename Key, typename Value>
inline std::size_t FlatMap<Key, Value>::size() const noexcept {
  return keys_.size();
}

template <typename Key, typename Value>
inline Key FlatMap<Key, Value>::nthKey(const std::size_t nth) const noexcept {
  assert(nth < keys_.size());
  return keys_[nth].first;
}

template <typename Key, typename Value>
inline const Value& FlatMap<Key, Value>::nthValue(const std::size_t nth) const noexcept {
  assert(nth < keys_.size());
  assert(keys_.size() == values_.size());
  assert(keys_[nth].second < values_.size());
  return values_[keys_[nth].second];
}

template <typename Key, typename Value>
inline Value& FlatMap<Key, Value>::nthValue(const std::size_t nth) noexcept {
  assert(nth < keys_.size());
  assert(keys_.size() == values_.size());
  assert(keys_[nth].second < values_.size());
  return values_[keys_[nth].second];
}

template <typename Key, typename Value>
inline const std::vector<Value>& FlatMap<Key, Value>::valuesVector() const noexcept {
  return values_;
}

template <typename Key, typename Value>
inline std::vector<Value>& FlatMap<Key, Value>::valuesVector() noexcept {
  return values_;
}

}  // namespace EvolutionNet

#endif  // EVOLUTION_NET_FLAT_MAP_HPP
