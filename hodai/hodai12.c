#include "nn.h"
#include <time.h>

// shuffle
void shuffle(int n, int *x){
  for(int i=0; i<n; i++){
    int j = rand()%n;
    x[j] = x[i];
  }
}

//test
int main(){
  srand(time(NULL));
  float *train_x = NULL;
  unsigned char *train_y = NULL;
  int train_count = -1;
  float *test_x = NULL;
  unsigned char * test_y = NULL;
  int test_count = -1;
  int width = -1;
  int height = -1;
  load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
  int *index = malloc(sizeof(int)*train_count);
  for(int i=0; i<train_count; i++){
    index[i]=i;
  }
  shuffle(train_count, index);
  for (int i=0; i<train_count; i++){
    printf("%d ", index[i]);
  }
  return 0;
}
