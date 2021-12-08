


namespace {

#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <array>


typedef double ElementType;

template <typename ElementType, size_t Rank>
struct Storage {
    ElementType *allocatedPtr;
    ElementType *alignedPtr;
    int64_t offset;
    std::array<int64_t, Rank> sizes;   // omitted when rank == 0
    std::array<int64_t, Rank> strides; // omitted when rank == 0

   template<typename... T>
   struct type { };

   template <typename... T>
   ElementType &operator()(T... arg) {
       return operator()(type<T...>(), arg...);
   }

   template <typename... T>
   const ElementType &operator()(T... arg) const {
       return operator()(type<T...>(), arg...);
   }

   template<typename... T>
   ElementType &operator()(type<T...>, T... arg);

   template<typename... T>
   const ElementType &operator()(type<T...>, T... arg) const;
};

typedef Storage<ElementType, 3> Storage3D;

Storage3D allocateStorage(const std::array<int64_t, 3> sizes) {
  Storage3D result;
  // initialize the size
  result.sizes[2] = sizes[0];
  result.sizes[1] = sizes[1];
  result.sizes[0] = sizes[2];
  // initialize the strides
  result.strides[2] = 1;
  result.strides[1] = result.sizes[2];
  result.strides[0] = result.sizes[2] * result.sizes[1];
  result.offset = halo_width * result.strides[0] +
                  halo_width * result.strides[1] +
                  halo_width * result.strides[2];
  const int64_t allocSize = sizes[0] * sizes[1] * sizes[2];
  result.allocatedPtr = new ElementType[allocSize + (32 - halo_width)];
  result.alignedPtr = &result.allocatedPtr[(32 - halo_width)];
  return result;
}

#endif // UTIL_H