#include "nn.h"

//Add
void add(int n, const float *x, float *o){
  for(int i=0; i<n; i++){
    o[i] += x[i];
  }
}

//Scale
void scale(int n, float x, float *o){
  for(int i=0; i<n; i++){
    o[i] *= x;
  }
}

//Init
void init(int n, float x, float *o){
  for(int i=0; i<n; i++){
    o[i] = x;
  }
}

//Rand_init
void rand_init(int n, float *o){
  for(int i=0; i<n; i++){
    o[i] = 2*rand()/RAND_MAX - 1;
  }
}
