#include <stdio.h>

void minmax(int data[], int *min, int *max){
  *min = data[0];
  *max = data[0];
  for(int i=0; i<3; i++){
    *min = (*min < data[i]) ? *min : data[i];
    *max = (*max > data[i]) ? *max : data[i];
  }
}

int main(void){
  int data[3];
  int min, max;
  printf("input 1st integer : ");
  scanf("%d", &data[0]);
  printf("input 2nd integer : ");
  scanf("%d", &data[1]);
  printf("input 3rd integer : ");
  scanf("%d", &data[2]);
  minmax(data, &min, &max);
  printf("min: %d, max: %d\n", min, max);
  return 0;
}
