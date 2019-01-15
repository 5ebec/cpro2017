#include "nn.h"

// Loss
float loss(const float *y, int t){
  return -log(y[t]+1e-7);
}
