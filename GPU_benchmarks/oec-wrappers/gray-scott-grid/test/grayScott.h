#ifndef CONTAINER_H
#define CONTAINER_H

void grayScott(const Storage3D& in, Storage3D& out) {
  for (int64_t i = 0; i < 256; ++i) {
    for (int64_t j = 0; j < 256; ++j) {
      for (int64_t k = 0; k < 256; ++k) {
        out(i, j, k) =
            -4.0 * in(i, j, k) + ((in(i - 1, j, k) + in(i + 1, j, k)) +
                                  (in(i, j + 1, k) + in(i, j - 1, k)));
      }
    }
  }
}

#endif